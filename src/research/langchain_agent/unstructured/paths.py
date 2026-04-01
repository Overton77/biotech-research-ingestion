from __future__ import annotations

from pathlib import Path

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_unstructured_dir(task_slug: str, *, root: Path = ROOT_FILESYSTEM) -> Path:
    return _ensure_dir(root / "runs" / task_slug / "unstructured")


def stage_candidate_dir(task_slug: str, *, root: Path = ROOT_FILESYSTEM) -> Path:
    return _ensure_dir(stage_unstructured_dir(task_slug, root=root) / "candidates")


def mission_unstructured_dir(mission_id: str, *, root: Path = ROOT_FILESYSTEM) -> Path:
    return _ensure_dir(root / "runs" / f"mission_{mission_id}" / "unstructured")


def mission_candidate_dir(mission_id: str, *, root: Path = ROOT_FILESYSTEM) -> Path:
    return _ensure_dir(mission_unstructured_dir(mission_id, root=root) / "candidates")


def isolated_validation_dir(validation_name: str, *, root: Path = ROOT_FILESYSTEM) -> Path:
    return _ensure_dir(root / "runs" / validation_name / "unstructured_validation")


def relative_to_root(path: Path, *, root: Path = ROOT_FILESYSTEM) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
