
import sys

# Set Windows event loop policy IMMEDIATELY, before anything else
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[sitecustomize] ✅ Windows SelectorEventLoop policy set (for psycopg compatibility)")
