from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.agent.subagent_types import (
    SUBAGENT_DESCRIPTIONS,
    VERCEL_AGENT_BROWSER_SUBAGENT,
)

_WIN_LONG_PREFIX = "\\\\?\\"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


load_dotenv(project_root() / ".env")


AGENT_RUNTIME_DIR = ROOT_FILESYSTEM / "agent"
AGENT_SKILLS_DIR = AGENT_RUNTIME_DIR / "skills"
AGENT_BROWSER_SKILL_DIR = AGENT_SKILLS_DIR / "agent-browser"


def agent_browser_skill_source_dir() -> Path:
    return project_root() / ".agents" / "skills" / "agent-browser"


def ensure_agent_browser_skill_workspace(
    *,
    root_filesystem: Path = ROOT_FILESYSTEM,
) -> Path:
    """Sync the project skill into the shared agent_outputs sandbox."""
    source = agent_browser_skill_source_dir()
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"agent-browser skill not found at {source}. "
            "Install it under .agents/skills/agent-browser before enabling this subagent."
        )

    runtime_dir = root_filesystem / "agent"
    skills_dir = runtime_dir / "skills"
    target = skills_dir / "agent-browser"
    skills_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def _normalize_for_containment(path: Path) -> Path:
    rendered = str(path)
    if rendered.startswith(_WIN_LONG_PREFIX):
        rendered = rendered[len(_WIN_LONG_PREFIX) :]
    return Path(rendered).resolve()


class WindowsPathSafeLocalShellBackend(LocalShellBackend):
    """Local shell backend with Windows-safe virtual path containment and UTF-8 execute output."""

    def _resolve_path(self, key: str) -> Path:
        if not self.virtual_mode:
            return super()._resolve_path(key)
        vpath = key if key.startswith("/") else "/" + key
        if ".." in vpath or vpath.startswith("~"):
            raise ValueError("Path traversal not allowed")
        full = (self.cwd / vpath.lstrip("/")).resolve()
        full_norm = _normalize_for_containment(full)
        cwd_norm = _normalize_for_containment(self.cwd)
        try:
            full_norm.relative_to(cwd_norm)
        except ValueError as exc:
            raise ValueError(
                f"Path:{full} outside root directory: {self.cwd}"
            ) from exc
        return full

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        try:
            result = subprocess.run(  # noqa: S602
                command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),
            )
            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout} seconds.",
                exit_code=124,
                truncated=False,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )


def _build_runtime_prompt() -> Any:
    @dynamic_prompt
    def _runtime_fragment(request: ModelRequest) -> str:
        state = request.state
        task_slug = state.get("task_slug", "unknown-task")
        mission_id = state.get("mission_id", "unknown-mission")
        run_dir = state.get("run_dir", f"runs/{task_slug}")
        artifact_dir = f"runs/{task_slug}/subagents/{VERCEL_AGENT_BROWSER_SUBAGENT}"
        handoff_path = f"{artifact_dir}/handoff.json"
        findings_path = f"{artifact_dir}/findings.md"

        return "\n".join(
            [
                "Run context:",
                f"- mission_id: {mission_id}",
                f"- task_slug: {task_slug}",
                f"- run_dir: {run_dir}",
                f"- artifact_dir: {artifact_dir}",
                f"- required_handoff_path: {handoff_path}",
                f"- suggested_findings_path: {findings_path}",
                "- Treat the filesystem root as the shared research sandbox.",
                "- Write only relative sandbox paths.",
            ]
        )

    return _runtime_fragment


def _build_system_prompt() -> str:
    return f"""
You are the Vercel agent-browser specialist subagent for biotech research.

Your job is to handle browser tasks that benefit from shell-driven browser automation via the
Vercel `agent-browser` CLI and its skill instructions.

Core operating rules:
- Read `/agent/skills/agent-browser/SKILL.md` before doing browser work whenever the request requires live browser automation.
- Use the built-in filesystem tools for reading and writing artifacts, and use the `execute` tool to run `agent-browser` shell commands.
- Prefer direct official pages and the most specific URL provided by the parent agent.
- Use a dedicated `--session` name for each delegated task and close the session before finishing.
- Prefer `agent-browser open`, `wait --load networkidle`, `snapshot -i`, `get text`, `get url`, and targeted follow-up actions over raw `curl`.
- Stay bounded. Do not browse aimlessly, do not loop on the same failed command, and record blockers instead of spiraling.

Artifact contract:
- Write all artifacts under `runs/<task_slug>/subagents/{VERCEL_AGENT_BROWSER_SUBAGENT}/`.
- Always write a machine-readable handoff file at `runs/<task_slug>/subagents/{VERCEL_AGENT_BROWSER_SUBAGENT}/handoff.json`.
- The handoff file must contain:
  {{
    "subagent_name": "{VERCEL_AGENT_BROWSER_SUBAGENT}",
    "summary": "concise description of what you produced",
    "artifacts": [{{"path": "relative/path", "description": "what the file contains"}}],
    "sources": ["url or identifier"],
    "errors": ["error text"]
  }}
- Also write a concise markdown findings artifact when you gathered evidence worth citing.

Final response requirements:
- Return a compact JSON object only, not prose.
- Include exactly these top-level keys:
  "subagent_name", "summary", "handoff_file", "artifact_paths", "notable_findings", "errors"
- "handoff_file" must point to the relative sandbox path for the handoff artifact.
- "artifact_paths" must include every file you created that the parent agent may want to read next.
""".strip()


def build_vercel_agent_browser_backend(
    *,
    root_filesystem: Path = ROOT_FILESYSTEM,
):
    ensure_agent_browser_skill_workspace(root_filesystem=root_filesystem)

    def factory(runtime: ToolRuntime[Any, Any]) -> CompositeBackend:
        shell_backend = WindowsPathSafeLocalShellBackend(
            root_dir=str(root_filesystem),
            virtual_mode=True,
            inherit_env=True,
        )
        return CompositeBackend(
            default=shell_backend,
            routes={
                "/memories/": StateBackend(runtime),
            },
        )

    return factory


def build_vercel_agent_browser_subagent(
    *,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver[Any],
    root_filesystem: Path = ROOT_FILESYSTEM,
) -> CompiledSubAgent:
    ensure_agent_browser_skill_workspace(root_filesystem=root_filesystem)
    model_name = os.getenv("VERCEL_AGENT_BROWSER_MODEL", "gpt-5")
    model = ChatOpenAI(model=model_name, temperature=0, use_responses_api=True)
    deep_agent = create_deep_agent(
        model=model,
        backend=build_vercel_agent_browser_backend(root_filesystem=root_filesystem),
        skills=["/agent/skills/"],
        middleware=[_build_runtime_prompt()],
        system_prompt=_build_system_prompt(),
        checkpointer=checkpointer,
        store=store,
        name=VERCEL_AGENT_BROWSER_SUBAGENT,
    ).with_config({"recursion_limit": 160})

    return CompiledSubAgent(
        name=VERCEL_AGENT_BROWSER_SUBAGENT,
        description=SUBAGENT_DESCRIPTIONS[VERCEL_AGENT_BROWSER_SUBAGENT],
        runnable=deep_agent,
    )
