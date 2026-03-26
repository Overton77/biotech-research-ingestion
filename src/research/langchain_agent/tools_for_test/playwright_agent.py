from __future__ import annotations

import os
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8932/mcp") 

DEFAULT_MODEL = os.getenv("BROWSER_AGENT_MODEL", "gpt-5.4-mini")
DEFAULT_BROWSER_MAX_TOOL_STEPS = int(os.getenv("BROWSER_AGENT_MAX_TOOL_STEPS", "20"))


def _stringify_message_content(content: object) -> str:
    """Convert LangChain message content into a plain string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    return str(content)


def _extract_final_text(agent_result: dict) -> str:
    """
    Pull the final assistant text out of a LangChain agent result payload.
    Falls back gracefully if the exact structure differs by version.
    """
    messages = agent_result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _stringify_message_content(msg.content)
            if text:
                return text

        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = _stringify_message_content(msg.get("content"))
            if text:
                return text

    return (
        "RESULTS\n"
        "No final browser-agent output was produced.\n\n"
        "SOURCES\n"
        "- none\n\n"
        "NOTES\n"
        "The inner browser agent completed without a parseable assistant message."
    )


def _build_browser_task_prompt(
    *,
    start_url: str,
    task_goal: str,
    success_criteria: str,
    hints: Optional[str],
    allowed_domains: Optional[list[str]],
    prefer_sources: bool,
    output_format: Literal["key_value", "bullets", "raw", "jsonish"],
    max_actions: int,
) -> str:
    domains_str = ""
    if allowed_domains:
        domains_str = (
            "\nAllowed domains (do not navigate elsewhere unless absolutely necessary):\n- "
            + "\n- ".join(allowed_domains)
        )

    hints_str = f"\nHints:\n{hints}" if hints else ""

    sources_req = (
        "\nInclude a SOURCES section with the exact URLs used."
        if prefer_sources
        else "\nSOURCES section is optional."
    )

    format_guidance = {
        "key_value": "Return RESULTS as one 'key: value' per line.",
        "bullets": "Return RESULTS as concise bullet points.",
        "raw": "Return RESULTS as plain text.",
        "jsonish": (
            'Return RESULTS as JSON-like text. Prefer double quotes and stable keys, '
            "but it does not need to be perfect JSON."
        ),
    }[output_format]

    return f"""
You are completing a browser interaction and extraction task.

Start URL:
{start_url}
{domains_str}

Task goal:
{task_goal}

Success criteria:
{success_criteria}

Behavior:
- First inspect the visible page and form a minimal plan.
- Use browser tools to navigate, click, scroll, expand accordions, open tabs/sections, and reveal hidden content as needed.
- Be efficient: stop once the success criteria are satisfied.
- Do not wander into unrelated pages or over-collect.

Constraints:
- Stay within roughly {max_actions} browser actions.
- If allowed domains are provided, stay within them.
- If the requested information cannot be found, say so clearly and briefly describe what you tried.

Output requirements:
- {format_guidance}
- Always include a header line exactly 'RESULTS'
- Always include a header line exactly 'SOURCES'
- Include a header line exactly 'NOTES' only if something remains uncertain or incomplete.
{sources_req}

{hints_str}
""".strip()


def _browser_agent_system_prompt() -> str:
    return (
        "You are a browser automation specialist.\n"
        "Use the available browser tools to inspect, navigate, click, scroll, expand sections, "
        "switch tabs, and extract the requested information.\n\n"
        "Work efficiently:\n"
        "- First inspect the visible page and form a minimal plan.\n"
        "- Use as few browser actions as possible.\n"
        "- Prefer actions that directly reveal the requested evidence.\n"
        "- Stop once the success criteria are satisfied.\n\n"
        "Rules:\n"
        "- Stay within allowed domains if provided.\n"
        "- Do not wander to unrelated pages.\n"
        "- If content is hidden behind tabs, accordions, 'show more', or technical details sections, reveal it.\n"
        "- If a browser interaction fails because an element reference is stale or a page snapshot changed, recover by "
        "re-inspecting the page or capturing a fresh snapshot before trying again.\n"
        "- If the requested evidence cannot be found, say so clearly and explain what you tried.\n"
        "- Return only the requested output format.\n\n"
        "Always return:\n"
        "1. RESULTS\n"
        "2. SOURCES\n"
        "3. NOTES (only if needed)"
    )


def _classify_browser_error(error_text: str) -> str:
    lowered = error_text.lower()

    if "not found in the current page snapshot" in lowered or "fresh snapshot" in lowered:
        return "stale_page_snapshot"

    if "net::err_aborted" in lowered or "page.goto" in lowered:
        return "navigation_aborted"

    if "timeout" in lowered:
        return "timeout"

    if "target closed" in lowered or "browser has been closed" in lowered:
        return "browser_closed"

    if "execution context was destroyed" in lowered:
        return "navigation_interrupted"

    return "browser_tool_error"


def _format_browser_error_result(
    *,
    start_url: str,
    task_goal: str,
    success_criteria: str,
    error_text: str,
) -> str:
    error_type = _classify_browser_error(error_text)

    retry_guidance = {
        "stale_page_snapshot": (
            "The page changed after an earlier snapshot. A follow-up attempt should first re-inspect "
            "the current page or capture a fresh snapshot before clicking again."
        ),
        "navigation_aborted": (
            "Navigation was interrupted or aborted. A follow-up attempt may need a simpler navigation path, "
            "a fresh page load, or fewer concurrent browser tasks."
        ),
        "timeout": (
            "The action likely timed out. A follow-up attempt may need fewer steps, a more direct target, "
            "or a page refresh before continuing."
        ),
        "browser_closed": (
            "The browser or target page closed unexpectedly. A follow-up attempt should restart from the start URL."
        ),
        "navigation_interrupted": (
            "The page changed during interaction. A follow-up attempt should re-read the current page state before acting."
        ),
        "browser_tool_error": (
            "A browser tool error occurred. A follow-up attempt should use a narrower, more explicit instruction set."
        ),
    }[error_type]

    return (
        "RESULTS\n"
        "Browser interaction task did not complete successfully.\n\n"
        "SOURCES\n"
        f"- {start_url}\n\n"
        "NOTES\n"
        f"- error_type: {error_type}\n"
        f"- task_goal: {task_goal}\n"
        f"- success_criteria: {success_criteria}\n"
        f"- raw_error: {error_text}\n"
        f"- suggested_retry_strategy: {retry_guidance}"
    )


async def run_playwright_mcp_agent(
    prompt: str,
    *,
    mcp_url: str = MCP_URL,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    max_tool_steps: int = DEFAULT_BROWSER_MAX_TOOL_STEPS,
) -> str:
    """
    Runs a LangChain create_agent browser agent backed by Playwright MCP tools.
    Returns the final assistant output as a plain string.
    """
    client = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "http",
                "url": mcp_url,
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt or _browser_agent_system_prompt(),
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config={"recursion_limit": max_tool_steps},
    )

    return _extract_final_text(result)


@tool(
    description=(
        "Use a browser-specialist LangChain agent backed by Playwright MCP tools to navigate "
        "a page, interact with it, and extract targeted information. Returns a parse-friendly string."
    ),
    parse_docstring=False,
)
async def browser_interaction_task(
    runtime: ToolRuntime,
    start_url: Annotated[str, "The exact page to start from."],
    task_goal: Annotated[str, "What the browser agent should accomplish."],
    success_criteria: Annotated[str, "What counts as a successful completion."],
    hints: Annotated[
        Optional[str],
        "Optional hints about which tabs, buttons, accordions, or page sections may matter.",
    ] = None,
    allowed_domains: Annotated[
        Optional[list[str]],
        "Optional allowlist of domains the browser agent may browse.",
    ] = None,
    output_format: Annotated[
        Literal["key_value", "bullets", "raw", "jsonish"],
        "Parse-friendly output style.",
    ] = "bullets",
    prefer_sources: Annotated[
        bool,
        "If True, require a SOURCES section with URLs used.",
    ] = True,
    max_actions: Annotated[
        int,
        "Soft cap for the inner browser agent's tool-using loop.",
    ] = 20,
) -> Command:
    prompt = _build_browser_task_prompt(
        start_url=start_url,
        task_goal=task_goal,
        success_criteria=success_criteria,
        hints=hints,
        allowed_domains=allowed_domains,
        prefer_sources=prefer_sources,
        output_format=output_format,
        max_actions=max_actions,
    )

    try:
        result_str = await run_playwright_mcp_agent(
            prompt,
            mcp_url=MCP_URL,
            model=DEFAULT_MODEL,
            max_tool_steps=max_actions,
        )
    except ToolException as e:
        result_str = _format_browser_error_result(
            start_url=start_url,
            task_goal=task_goal,
            success_criteria=success_criteria,
            error_text=str(e),
        )
    except Exception as e:
        result_str = _format_browser_error_result(
            start_url=start_url,
            task_goal=task_goal,
            success_criteria=success_criteria,
            error_text=f"{type(e).__name__}: {e}",
        )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result_str,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )