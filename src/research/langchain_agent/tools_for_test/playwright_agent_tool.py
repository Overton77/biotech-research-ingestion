from __future__ import annotations

from langchain.tools import tool, ToolRuntime  
from typing import Optional, Literal, Union, List, Tuple, Annotated, Dict, Any 
import os
import asyncio
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from langgraph.types import Command
from langchain.messages import ToolMessage 



load_dotenv() 

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8932/mcp") 


DEFAULT_MODEL = "gpt-5-mini"

async def run_playwright_mcp_agent(
    prompt: str,
    *,
    mcp_url: str = MCP_URL,
    model: str = "gpt-5",
    system_prompt: str | None = None,
) -> str:
    """Runs a LlamaIndex FunctionAgent backed by Playwright MCP tools. Returns plain string."""
    mcp_client = BasicMCPClient(mcp_url)
    tool_spec = McpToolSpec(client=mcp_client)
    tools = await tool_spec.to_tool_list_async()

    llm = OpenAI(model=model)

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt
        or (
            "You are a browser automation agent. Use the browser tools to navigate, click, scroll, "
            "open spec sections, and extract the requested information. "
            "Return ONLY the final answer in the requested output format."
        ),
    )

    response = await agent.run(prompt)
    return str(response)



def _build_spec_prompt(
    *,
    exact_url: str,
    output_instructions: str,
    scraping_hints: Optional[str],
    allowed_domains: Optional[List[str]],
    prefer_sources: bool,
    output_format: Literal["key_value", "bullets", "raw", "jsonish"],
    max_actions: int,
) -> str:
    domains_str = ""
    if allowed_domains:
        domains_str = (
            "\nAllowed domains (do not navigate elsewhere):\n- "
            + "\n- ".join(allowed_domains)
        )

    hints_str = f"\nScraping hints:\n{scraping_hints}" if scraping_hints else ""

    sources_req = (
        "\nInclude a SOURCES section with the URLs you used."
        if prefer_sources
        else "\nSOURCES section is optional."
    )

    format_guidance = {
        "key_value": "Return RESULTS as one 'key: value' per line.",
        "bullets": "Return RESULTS as bullet points.",
        "raw": "Return RESULTS as plain text.",
        "jsonish": (
            "Return RESULTS as JSON-like text (double quotes preferred), but it can be imperfect JSON."
        ),
    }[output_format]

    return f"""
You are extracting product specifications from the web.

Start here:
{exact_url}
{domains_str}

Goal:
{output_instructions}

Behavior:
- Use browser tools to navigate, click 'View full specs' / 'Technical specifications' / 'Details', expand accordions, and scroll if needed.
- If specs are split across tabs/sections, gather them.
- Stop once you have the requested outputs. Do not over-collect.

Constraints:
- Keep within ~{max_actions} browser actions worth of work (be efficient).

Output requirements:
- {format_guidance}
- Always include a header line 'RESULTS' then the results.
{sources_req}

{hints_str}
""".strip()


@tool(
    description="Use LlamaIndex FunctionAgent + Playwright MCP to navigate a page and extract product specs. Returns a parse-friendly string.",
    parse_docstring=False,
)
async def playwright_mcp_specs(
    runtime: ToolRuntime,
    exact_url: Annotated[str, "The exact page to start from (product page or spec sheet)."],
    output_instructions: Annotated[str, "What to extract and how detailed (fields, units, variants, etc.)."],
    scraping_hints: Annotated[Optional[str], "Optional hints (e.g. which tab/accordion, 'View more specs', etc.)."] = None,
    allowed_domains: Annotated[Optional[List[str]], "Optional allowlist of domains the agent may browse."] = None,
    output_format: Annotated[Literal["key_value", "bullets", "raw", "jsonish"], "Parse-friendly output style (still a string)."] = "key_value",
    prefer_sources: Annotated[bool, "If True, require SOURCES urls for verification."] = True,
    max_actions: Annotated[int, "Soft cap to keep browsing bounded."] = 25,
) -> Command:

    prompt = _build_spec_prompt(
        exact_url=exact_url,
        output_instructions=output_instructions,
        scraping_hints=scraping_hints,
        allowed_domains=allowed_domains,
        prefer_sources=prefer_sources,
        output_format=output_format,
        max_actions=max_actions,
    )

    result_str = await run_playwright_mcp_agent(
        prompt,
        mcp_url=MCP_URL,
        model=DEFAULT_MODEL,
    ) 

    return Command( 
        update={ 
            "messages": [ToolMessage(content=result_str, tool_call_id=runtime.tool_call_id)],
        }
    )

    