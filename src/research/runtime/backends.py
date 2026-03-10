"""Workspace path builders and backend factories for research missions.

Path layout:
  /tmp/research_missions/{mission_id}/tasks/{task_id}/
      workspace/
      outputs/
      scratch/
      subagents/{subagent_name}/
          workspace/
          outputs/
          scratch/
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from deepagents.backends import FilesystemBackend


def mission_root(mission_id: str) -> Path:
    return Path("/tmp/research_missions") / mission_id


def task_root(mission_id: str, task_id: str) -> Path:
    return mission_root(mission_id) / "tasks" / task_id


def subagent_root(mission_id: str, task_id: str, subagent_name: str) -> Path:
    return task_root(mission_id, task_id) / "subagents" / subagent_name


async def ensure_task_workspace(mission_id: str, task_id: str) -> Path:
    """Create workspace directories for a task. Returns task root path."""
    root = task_root(mission_id, task_id)
    for subdir in ("workspace", "outputs", "scratch"):
        target = root / subdir
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    return root


async def ensure_subagent_workspace(
    mission_id: str, task_id: str, subagent_name: str
) -> Path:
    """Create workspace directories for a subagent. Returns subagent root."""
    root = subagent_root(mission_id, task_id, subagent_name)
    for subdir in ("workspace", "outputs", "scratch"):
        target = root / subdir
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    return root


def build_task_backend(
    mission_id: str,
    task_id: str,
    store: Any,
) -> Callable:
    """
    Return a CompositeBackend factory (callable that takes runtime).
    Routes /memories/ → StoreBackend(runtime), everything else → FilesystemBackend.
    """
    root = task_root(mission_id, task_id)

    def backend_factory(runtime: Any) -> Any:
        from deepagents.backends import StoreBackend
        from deepagents.backends.composite import CompositeBackend
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(root)),
            routes={"/memories/": StoreBackend(runtime)},
        )

    return backend_factory


def build_subagent_backend(
    mission_id: str,
    task_id: str,
    subagent_name: str,
) -> FilesystemBackend:
    """
    Return a plain FilesystemBackend for a subagent.
    Subagents get their own isolated workspace, not a composite backend.
    """
    root = subagent_root(mission_id, task_id, subagent_name)
    return FilesystemBackend(root_dir=str(root))
