"""Workspace path builders and backend factories for research missions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable

from langgraph.runtime import Runtime

from deepagents.backends import FilesystemBackend


# Windows long-path prefix; pathlib can produce it on resolve(), breaking relative_to()
_WIN_LONG_PREFIX = "\\\\?\\"


def _normalize_for_containment(p: Path) -> Path:
    """Normalize path so containment checks work on Windows (strip long-path prefix)."""
    s = str(p)
    if s.startswith(_WIN_LONG_PREFIX):
        s = s[len(_WIN_LONG_PREFIX) :]
    return Path(s).resolve()


def _default_workspace_root() -> Path:
    # Adjust parents[N] to match your actual project layout.
    # Example assumes this file lives somewhere like:
    # src/.../workspace.py  -> project root is parents[2] or parents[3]
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".deepagents" / "research_missions"


def workspace_root() -> Path:
    override = os.environ.get("DEEP_AGENTS_WORKSPACE_ROOT")
    root = Path(override).expanduser().resolve() if override else _default_workspace_root().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def mission_root(mission_id: str) -> Path:
    return workspace_root() / mission_id


def task_root(mission_id: str, task_id: str) -> Path:
    return mission_root(mission_id) / "tasks" / task_id


def subagent_root(mission_id: str, task_id: str, subagent_name: str) -> Path:
    return task_root(mission_id, task_id) / "subagents" / subagent_name


async def ensure_task_workspace(mission_id: str, task_id: str) -> Path:
    root = task_root(mission_id, task_id)
    for subdir in ("workspace", "outputs", "scratch"):
        target = root / subdir
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    return root


async def ensure_subagent_workspace(
    mission_id: str, task_id: str, subagent_name: str
) -> Path:
    root = subagent_root(mission_id, task_id, subagent_name)
    for subdir in ("workspace", "outputs", "scratch"):
        target = root / subdir
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    return root


class WindowsPathSafeFilesystemBackend(FilesystemBackend):
    """
    FilesystemBackend that normalizes paths before containment checks so that
    virtual_mode works on Windows when resolve() returns a long-path (\\\\?\\)
    prefix and self.cwd does not, which would otherwise make relative_to() fail.
    """

    def _resolve_path(self, key: str) -> Path:
        if not self.virtual_mode:
            return super()._resolve_path(key)
        vpath = key if key.startswith("/") else "/" + key
        if ".." in vpath or vpath.startswith("~"):
            raise ValueError("Path traversal not allowed")
        full = (self.cwd / vpath.lstrip("/")).resolve()
        # Normalize so containment check works on Windows (long-path prefix)
        full_norm = _normalize_for_containment(full)
        cwd_norm = _normalize_for_containment(self.cwd)
        try:
            full_norm.relative_to(cwd_norm)
        except ValueError:
            raise ValueError(
                f"Path:{full} outside root directory: {self.cwd}"
            ) from None
        return full


def build_task_backend(
    mission_id: str,
    task_id: str,
) -> Callable[[Any], Any]:
    """
    Composite backend:
    - filesystem for workspace/output/scratch (Windows path-safe)
    - store-backed persistence for /memories/
    """
    root = task_root(mission_id, task_id).resolve()

    def backend_factory(runtime: Runtime) -> Any:
        from deepagents.backends import StoreBackend
        from deepagents.backends.composite import CompositeBackend

        return CompositeBackend(
            default=WindowsPathSafeFilesystemBackend(
                root_dir=str(root),
                virtual_mode=True,
            ),
            routes={
                "/memories/": StoreBackend(runtime),
            },
        )

    return backend_factory


def build_subagent_backend(
    mission_id: str,
    task_id: str,
    subagent_name: str,
) -> WindowsPathSafeFilesystemBackend:
    """
    Isolated filesystem backend for a subagent (Windows path-safe).
    """
    root = subagent_root(mission_id, task_id, subagent_name).resolve()
    return WindowsPathSafeFilesystemBackend(
        root_dir=str(root),
        virtual_mode=True,
    )