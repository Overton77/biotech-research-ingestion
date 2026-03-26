from __future__ import annotations

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM


FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List directories and files inside the shared sandbox. "
        "Use relative paths like runs, reports, and scratch."
    ),
    "read_file": (
        "Read saved notes, intermediate findings, and handoff artifacts from "
        "sandbox-relative paths."
    ),
    "write_file": (
        "Create structured intermediate files and final reports inside the shared "
        "sandbox only. Prefer markdown or json."
    ),
    "edit_file": (
        "Update an existing findings file, handoff artifact, or report incrementally "
        "inside the shared sandbox."
    ),
}

FILESYSTEM_SYSTEM_PROMPT = (
    "Use the filesystem aggressively for intermediate research state. "
    "All paths must stay inside the sandbox root. "
    "Use only relative sandbox paths such as runs/, reports/, and scratch/. "
    "Never use absolute host paths."
)

filesystem_backend = FilesystemBackend(
    root_dir=str(ROOT_FILESYSTEM),
    virtual_mode=True,
)


def build_shared_filesystem_middleware(
    *,
    backend: FilesystemBackend = filesystem_backend,
) -> FilesystemMiddleware:
    return FilesystemMiddleware(
        backend=backend,
        system_prompt=FILESYSTEM_SYSTEM_PROMPT,
        custom_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
    )
