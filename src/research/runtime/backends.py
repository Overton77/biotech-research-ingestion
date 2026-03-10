"""Workspace path builders and backend factories for research missions."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable

from deepagents.backends import FilesystemBackend


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


def build_task_backend(
    mission_id: str,
    task_id: str,
) -> Callable[[Any], Any]:
    """
    Composite backend:
    - filesystem for workspace/output/scratch
    - store-backed persistence for /memories/
    """
    root = task_root(mission_id, task_id).resolve()

    def backend_factory(runtime: Any) -> Any:
        from deepagents.backends import StoreBackend
        from deepagents.backends.composite import CompositeBackend

        return CompositeBackend(
            default=FilesystemBackend(
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
) -> FilesystemBackend:
    """
    Isolated filesystem backend for a subagent.
    """
    root = subagent_root(mission_id, task_id, subagent_name).resolve()
    return FilesystemBackend(
        root_dir=str(root),
        virtual_mode=True,
    )