# src/infrastructure/runtime/windows_asyncio.py
from __future__ import annotations

import sys


def configure_windows_asyncio() -> None:
    if sys.platform.startswith("win"):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())