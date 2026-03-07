"""Application runner.

Sets ``WindowsSelectorEventLoopPolicy`` on Windows *before* uvicorn creates
the event loop, which is required for psycopg3 / AsyncPostgresSaver.

Usage:
    uv run python main.py
    # or in production:
    uv run python main.py --host 0.0.0.0 --port 8001
"""

import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
