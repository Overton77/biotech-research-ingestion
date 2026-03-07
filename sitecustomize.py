"""
Site Customization for Windows Event Loop Policy

This file is automatically imported by Python before any other code runs.
It ensures that the Windows event loop policy is set to WindowsSelectorEventLoopPolicy
BEFORE the Taskiq CLI or any other code creates an event loop.

This is the most reliable way to fix the psycopg ProactorEventLoop incompatibility on Windows.

See: https://docs.python.org/3/library/site.html#module-sitecustomize
"""
import sys

# Set Windows event loop policy IMMEDIATELY, before anything else
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[sitecustomize] ✅ Windows SelectorEventLoop policy set (for psycopg compatibility)")
