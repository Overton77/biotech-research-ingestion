from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from src.research.langchain_agent.agent.factory import browser_control_agent

load_dotenv()


def _print_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n{'=' * 24} {label} {'=' * 24}\n")
    for message in result.get("messages", []):
        role = getattr(message, "type", None) or message.__class__.__name__
        content = getattr(message, "content", "")
        print(f"[{role}]")
        print(content)
        print()


async def run_prompt(label: str, prompt: str) -> dict[str, Any]:
    result = await browser_control_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )
    _print_result(label, result)
    return result


async def test_parallel_browser_tasks() -> None:
    """
    Test 1:
    Ask the browser-control subagent to handle two browser tasks in parallel.
    The prompt encourages it to launch separate browser interactions for two product pages.
    """
    prompt = """
You are coordinating browser work.

Run two browser interaction tasks in parallel if possible.

Task A:
- Start URL: https://www.qualialife.com/shop/qualia-mind
- Goal: Extract the page title, product name, top-level promise/value proposition, and whether the page visibly includes ingredient or supplement facts sections.
- Success criteria: Return concise bullet points and include the exact URL used.

Task B:
- Start URL: https://www.qualialife.com/shop/qualia-nad
- Goal: Extract the page title, product name, top-level promise/value proposition, and whether the page visibly includes ingredient or supplement facts sections.
- Success criteria: Return concise bullet points and include the exact URL used.

Requirements:
- Use browser_interaction_task for both tasks.
- Prefer parallel execution if your runtime supports it.
- After both tasks finish, provide a short comparison section highlighting any obvious differences in positioning or page structure.
- Include sources.
""".strip()

    await run_prompt("TEST 1 - PARALLEL BROWSER TASKS", prompt)


async def test_two_stage_navigation_then_deep_dive() -> None:
    """
    Test 2:
    First use the browser for navigation/discovery on the shop page,
    then run it again to go deeper on one identified product page.
    """
    prompt = """
You are coordinating browser work.

You may use the browser interaction tool multiple times.

Stage 1: navigation / identification
- Start URL: https://www.qualialife.com/shop
- Goal: Identify one product page on qualialife.com that appears to focus most directly on cognitive performance, focus, mental clarity, or brain health.
- Success criteria:
  - Name the most relevant product you found.
  - Give the exact product URL.
  - Briefly explain why it appears to be the best match.
  - Stay on qualialife.com.

Stage 2: go deeper
- After identifying the best-matching product page, run the browser interaction tool again on that exact product page.
- Goal: Extract the core positioning/benefit claims, and determine whether there are visible sections for ingredients, supplement facts, or reviews.
- Success criteria:
  - Return concise bullet points.
  - Include the exact URLs used.
  - Note any section labels or clickable areas that appear relevant.

Final answer requirements:
- Separate your answer into:
  1. IDENTIFIED TARGET
  2. DEEPER FINDINGS
  3. SOURCES
""".strip()

    await run_prompt("TEST 2 - NAVIGATION THEN DEEP DIVE", prompt)


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not os.getenv("MCP_URL"):
        print("MCP_URL not set; defaulting to http://127.0.0.1:8932/mcp")

    # Run both top-level tests sequentially so the logs stay readable.
    await test_parallel_browser_tasks()
    await test_two_stage_navigation_then_deep_dive()


if __name__ == "__main__":
    asyncio.run(main())