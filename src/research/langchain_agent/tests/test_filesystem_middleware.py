from __future__ import annotations

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.tools.middleware.filesystem import (
    _normalize_sandbox_path,
)


def test_normalize_sandbox_path_converts_absolute_rooted_path_to_relative():
    absolute = str(ROOT_FILESYSTEM / "runs" / "example" / "notes.md")
    normalized = _normalize_sandbox_path(absolute)
    assert normalized == "runs/example/notes.md"


def test_normalize_sandbox_path_preserves_path_outside_root():
    outside = "C:/temp/example.md"
    assert _normalize_sandbox_path(outside) == outside
