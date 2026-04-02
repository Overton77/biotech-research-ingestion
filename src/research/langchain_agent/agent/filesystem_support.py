from __future__ import annotations

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM
from src.research.langchain_agent.agent.prompts.shared_filesystem_prompts import FILESYSTEM_TOOL_DESCRIPTIONS, FILESYSTEM_SYSTEM_PROMPT




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
