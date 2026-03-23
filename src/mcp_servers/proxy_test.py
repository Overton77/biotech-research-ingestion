# research_agent/human_upgrade/tools/integrations/playwright_proxy.py
import asyncio
import os

from fastmcp import FastMCP 
from fastmcp.server import create_proxy 


biotech_proxy_mcp = FastMCP("BiotechProxy") 


# playwright_config = {
#     "mcpServers": {
#         "playwright": {
#             "command": "npx",
#             "args": ["-y", "@playwright/mcp@latest"]
#         }
#     }
# }

# biotech_proxy_mcp.mount(create_proxy(playwright_config), namespace="playwright")

biotech_proxy_mcp.mount(create_proxy("https://paper-search-mcp-openai-v2--titansneaker.run.tools"), namespace="medpapers")


async def main(): 
    await biotech_proxy_mcp.run_async(transport="http", host="127.0.0.1", port=8922)
    print(f"Biotech Proxy MCP server running on http://127.0.0.1:8922/mcp")


if __name__ == "__main__":
    asyncio.run(main())
