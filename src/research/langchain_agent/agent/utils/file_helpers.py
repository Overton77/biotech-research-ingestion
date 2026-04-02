import asyncio
from pathlib import Path
import aiofiles

from src.research.langchain_agent.agent.constants import RUNS_DIR, REPORTS_DIR, SCRATCH_DIR, ROOT_FILESYSTEM

async def list_agent_files(root: Path = ROOT_FILESYSTEM) -> list[str]:
    def _scan() -> list[str]:
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
        )

    return await asyncio.to_thread(_scan)


async def print_agent_files(root: Path = ROOT_FILESYSTEM) -> None:
    files = await list_agent_files(root)
    print("\n=== AGENT FILES ===")
    if not files:
        print("(no files)")
        return
    for f in files:
        print(f"- {f}")


async def read_file_text(path_relative: str, root: Path = ROOT_FILESYSTEM) -> str:
    path = root / path_relative
    if not await asyncio.to_thread(path.exists):
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def dump_file(path_relative: str, root: Path = ROOT_FILESYSTEM) -> None:
    text = await read_file_text(path_relative, root=root)
    print(f"\n=== {path_relative} ===")
    if not text:
        print("(missing)")
        return
    print(text)


async def ensure_dirs() -> None:
    for p in (ROOT_FILESYSTEM, RUNS_DIR, REPORTS_DIR, SCRATCH_DIR):
        await asyncio.to_thread(p.mkdir, parents=True, exist_ok=True)