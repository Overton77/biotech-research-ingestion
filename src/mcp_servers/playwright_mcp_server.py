import asyncio
import os

from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy


def build_proxy():
    backend = StdioTransport(
        command="npx",
        args=["@playwright/mcp@latest", "--isolated"],
        env=os.environ.copy(),
    )
    return create_proxy(backend, name="playwright-proxy")


async def main():
    proxy = build_proxy()
    await proxy.run_async(transport="http", host="127.0.0.1", port=8932)


if __name__ == "__main__":
    asyncio.run(main())