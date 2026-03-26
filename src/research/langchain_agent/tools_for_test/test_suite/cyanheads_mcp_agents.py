from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, ToolCallInterceptor
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DEFAULT_MODEL_NAME = "gpt-5-mini"
memory_saver = MemorySaver()


@dataclass(frozen=True)
class ServerSpec:
    name: str
    url: str
    system_prompt: str
    default_test_prompt: str
    allowed_tool_names: tuple[str, ...] | None = None


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Tool used or tool family consulted.")
    status: str = Field(description="Short status like success, partial, failed, fallback.")
    summary: str = Field(description="Compact description of what the tool revealed.")
    identifiers: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class ResearchSubagentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: str = Field(description="Which MCP-backed subagent produced this result.")
    query_focus: str = Field(description="What the subagent believes it was asked to find.")
    concise_answer: str = Field(description="Short actionable answer for a coordinator agent.")
    observations: list[ToolObservation] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    fallback_actions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25
    retryable_markers: tuple[str, ...] = (
        "429",
        "too many requests",
        "rate limit",
        "rate-limit",
        "timeout",
        "timed out",
        "503",
        "502",
        "504",
        "500",
        "brokenresourceerror",
        "temporarily unavailable",
        "connection reset",
    )


class RateLimitRetryInterceptor:
    """Retry MCP tool calls for transient server and transport failures."""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        server_allowlist: Sequence[str] | None = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self.server_allowlist = tuple(server_allowlist) if server_allowlist else None

    def _should_retry_server(self, server_name: str) -> bool:
        return self.server_allowlist is None or server_name in self.server_allowlist

    def _is_retryable(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in self.policy.retryable_markers)

    async def __call__(self, request: MCPToolCallRequest, handler) -> Any:
        if not self._should_retry_server(request.server_name):
            return await handler(request)

        last_exc: BaseException | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                return await handler(request)
            except Exception as exc:  # pragma: no cover - exercised in live mode
                last_exc = exc
                if attempt >= self.policy.max_attempts or not self._is_retryable(exc):
                    raise
                delay = min(
                    self.policy.max_delay_seconds,
                    self.policy.base_delay_seconds * (2 ** (attempt - 1)),
                )
                delay += random.uniform(0.0, self.policy.jitter_seconds)
                await asyncio.sleep(delay)

        raise RuntimeError(f"Retry interceptor exhausted without result: {last_exc}")


SERVER_SPECS: dict[str, ServerSpec] = {
    "pubmed": ServerSpec(
        name="pubmed",
        url="https://pubmed.caseyjhand.com/mcp",
        system_prompt=(
            "You are a PubMed literature subagent for biotech entity research. "
            "Use PubMed-native tools for literature discovery first. Search broadly when the "
            "entity or mechanism is ambiguous; fetch article details directly when PMIDs are known. "
            "Use MeSH lookups to refine biomedical terms, use full-text retrieval only when needed, "
            "and explicitly say when only abstract-level evidence is available. Avoid unnecessary tool "
            "calls once enough evidence has been gathered. Return structured findings that a coordinator "
            "can map to companies, compounds, people, and trials."
        ),
        default_test_prompt=(
            "Find recent papers on partial cellular reprogramming in mice. "
            "Summarize the main themes, note whether full text was available, and surface the most "
            "useful PMIDs for downstream extraction."
        ),
        allowed_tool_names=(
            "pubmed_search_articles",
            "pubmed_fetch_articles",
            "pubmed_fetch_fulltext",
            "pubmed_format_citations",
            "pubmed_find_related",
            "pubmed_lookup_mesh",
            "pubmed_spell_check",
        ),
    ),
    "pubchem": ServerSpec(
        name="pubchem",
        url="https://pubchem.caseyjhand.com/mcp",
        system_prompt=(
            "You are a PubChem compound intelligence subagent for biotech entity research. "
            "Resolve compounds carefully, prefer direct retrieval when a CID is known, and inspect "
            "properties, safety, xrefs, and bioactivity only to the depth needed for the question. "
            "Be explicit when a compound lookup is ambiguous or when evidence is indirect. Return "
            "identifiers and URLs that can be linked to downstream trial and company research."
        ),
        default_test_prompt=(
            "Analyze rapamycin in PubChem. Return the key compound identifiers, physicochemical "
            "properties, safety findings, and any bioactivity context useful for linking the compound "
            "to company and trial research."
        ),
        allowed_tool_names=(
            "pubchem_search_compounds",
            "pubchem_get_compound_details",
            "pubchem_get_compound_safety",
            "pubchem_get_compound_xrefs",
            "pubchem_get_bioactivity",
            "pubchem_search_assays",
            "pubchem_get_summary",
            "pubchem_get_compound_image",
        ),
    ),
    "clinicaltrials": ServerSpec(
        name="clinicaltrials",
        url="https://clinicaltrials.caseyjhand.com/mcp",
        system_prompt=(
            "You are a ClinicalTrials.gov subagent for biotech entity research. "
            "Search efficiently with narrow queries and limited page sizes. If an NCT ID is known, "
            "retrieve the study directly before doing broader searches. Prefer structured API detail "
            "from clinicaltrials_get_study when it contains the needed protocol sections. If detail is "
            "missing, incomplete, or rate limited, return the canonical ClinicalTrials.gov study URL and "
            "recommend a fallback download-and-markdown path. Treat 429s as transient, avoid overcalling "
            "the API, and preserve missing-field details for downstream logic."
        ),
        default_test_prompt=(
            "Find interventional clinical trials related to partial cellular reprogramming, "
            "epigenetic rejuvenation, or closely related age-reversal approaches. If direct matches are "
            "sparse, expand to adjacent regeneration or longevity concepts and explain any retrieval gaps."
        ),
        allowed_tool_names=(
            "clinicaltrials_search_studies",
            "clinicaltrials_get_study",
            "clinicaltrials_get_study_results",
            "clinicaltrials_get_field_values",
            "clinicaltrials_analyze_trends",
            "clinicaltrials_compare_studies",
            "clinicaltrials_find_eligible_studies",
        ),
    ),
}


def default_retry_interceptors() -> list[ToolCallInterceptor]:
    return [
        RateLimitRetryInterceptor(
            server_allowlist=("clinicaltrials", "pubchem", "pubmed"),
        )
    ]


def stringify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    inner = getattr(exc, "exceptions", None)
    if inner:
        inner_text = " | ".join(f"{type(item).__name__}: {item}" for item in inner)
        return f"{text} [{inner_text}]"
    return text


def extract_tool_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result and isinstance(result["text"], str):
            return result["text"]
        return json.dumps(result, indent=2, default=str)
    if isinstance(result, list):
        blocks: list[str] = []
        for item in result:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                blocks.append(item["text"])
            else:
                blocks.append(str(item))
        return "\n\n".join(blocks)
    return str(result)


def extract_final_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    if not messages:
        return ""
    final_message = messages[-1]
    content = getattr(final_message, "content", final_message)
    if isinstance(content, str):
        return content
    return extract_tool_text(content)


def parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def clinicaltrials_study_url(nct_id: str) -> str:
    return f"https://clinicaltrials.gov/study/{nct_id}"


def assess_clinicaltrials_detail_payload(payload: dict[str, Any]) -> dict[str, Any]:
    protocol = payload.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    design = protocol.get("designModule") or {}
    study_type = design.get("studyType")
    nct_id = identification.get("nctId")

    common_sections = (
        "identificationModule",
        "statusModule",
        "sponsorCollaboratorsModule",
        "descriptionModule",
        "conditionsModule",
        "eligibilityModule",
    )
    interventional_sections = ("armsInterventionsModule", "outcomesModule")

    missing_sections = [section for section in common_sections if not protocol.get(section)]
    if study_type == "INTERVENTIONAL":
        missing_sections.extend(
            section for section in interventional_sections if not protocol.get(section)
        )

    direct_retrieval_sufficient = bool(nct_id) and not missing_sections
    fallback_actions: list[str] = []
    if not direct_retrieval_sufficient:
        fallback_actions.append(
            "Use the canonical ClinicalTrials.gov study URL and preserve the missing sections."
        )
        fallback_actions.append(
            "If downstream extraction needs the missing content, download the study page or file and convert it to markdown."
        )

    return {
        "nct_id": nct_id,
        "canonical_url": clinicaltrials_study_url(nct_id) if nct_id else None,
        "study_type": study_type,
        "missing_sections": missing_sections,
        "direct_retrieval_sufficient": direct_retrieval_sufficient,
        "fallback_actions": fallback_actions,
    }


def _headers_from_env(server_name: str) -> dict[str, str]:
    """Allow optional bearer auth per server via env vars."""
    specific = f"{server_name.upper()}_MCP_BEARER_TOKEN"
    token = os.getenv(specific) or os.getenv("MCP_BEARER_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def make_client(
    server_spec: ServerSpec,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> MultiServerMCPClient:
    config: dict[str, Any] = {
        server_spec.name: {
            "transport": "http",
            "url": server_spec.url,
        }
    }
    headers = _headers_from_env(server_spec.name)
    if headers:
        config[server_spec.name]["headers"] = headers
    return MultiServerMCPClient(
        config,
        tool_interceptors=list(tool_interceptors or []),
        tool_name_prefix=False,
    )


async def get_server_tools(
    server_spec: ServerSpec,
    allowed_tool_names: Iterable[str] | None = None,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
):
    client = make_client(server_spec, tool_interceptors=tool_interceptors)
    all_tools = await client.get_tools(server_name=server_spec.name)
    allowed = set(allowed_tool_names or server_spec.allowed_tool_names or [])
    if not allowed:
        return all_tools
    return [tool for tool in all_tools if tool.name in allowed]


async def get_server_prompt_metadata(
    server_spec: ServerSpec,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    client = make_client(server_spec, tool_interceptors=tool_interceptors)
    try:
        async with client.session(server_spec.name) as session:
            result = await session.list_prompts()
        prompts = []
        for prompt in result.prompts:
            prompts.append(
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": [
                        {
                            "name": argument.name,
                            "description": argument.description,
                            "required": argument.required,
                        }
                        for argument in prompt.arguments or []
                    ],
                }
            )
        return {"server": server_spec.name, "prompts": prompts}
    except Exception as exc:  # pragma: no cover - live network behavior
        return {"server": server_spec.name, "error": stringify_exception(exc), "prompts": []}


async def get_server_resource_metadata(
    server_spec: ServerSpec,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    client = make_client(server_spec, tool_interceptors=tool_interceptors)
    try:
        async with client.session(server_spec.name) as session:
            result = await session.list_resources()
        resources = []
        for resource in result.resources:
            resources.append(
                {
                    "name": resource.name,
                    "title": resource.title,
                    "uri": str(resource.uri),
                    "description": resource.description,
                    "mime_type": resource.mimeType,
                }
            )
        return {"server": server_spec.name, "resources": resources}
    except Exception as exc:  # pragma: no cover - live network behavior
        return {"server": server_spec.name, "error": stringify_exception(exc), "resources": []}


async def fetch_server_prompt(
    server_spec: ServerSpec,
    prompt_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    client = make_client(server_spec, tool_interceptors=tool_interceptors)
    messages = await client.get_prompt(
        server_name=server_spec.name,
        prompt_name=prompt_name,
        arguments=arguments,
    )
    return {
        "server": server_spec.name,
        "prompt_name": prompt_name,
        "message_count": len(messages),
        "messages": [
            {
                "type": type(message).__name__,
                "content": getattr(message, "content", str(message)),
            }
            for message in messages
        ],
    }


async def fetch_server_resources(
    server_spec: ServerSpec,
    *,
    uris: str | list[str] | None = None,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    client = make_client(server_spec, tool_interceptors=tool_interceptors)
    blobs = await client.get_resources(server_name=server_spec.name, uris=uris)
    return {
        "server": server_spec.name,
        "resource_count": len(blobs),
        "resources": [
            {
                "mime_type": blob.mimetype,
                "source": blob.source,
                "text": blob.as_string(),
            }
            for blob in blobs
        ],
    }


async def inspect_server_capabilities(
    server_spec: ServerSpec,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    tools_summary = await smoke_test_tool_discovery(
        server_spec,
        tool_interceptors=tool_interceptors,
    )
    prompt_summary = await get_server_prompt_metadata(
        server_spec,
        tool_interceptors=tool_interceptors,
    )
    resource_summary = await get_server_resource_metadata(
        server_spec,
        tool_interceptors=tool_interceptors,
    )
    return {
        **tools_summary,
        "prompts": prompt_summary.get("prompts", []),
        "prompt_error": prompt_summary.get("error"),
        "resources": resource_summary.get("resources", []),
        "resource_error": resource_summary.get("error"),
    }


async def invoke_server_tool(
    server_spec: ServerSpec,
    tool_name: str,
    payload: dict[str, Any],
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    tools = await get_server_tools(server_spec, tool_interceptors=tool_interceptors)
    selected_tool = next(tool for tool in tools if tool.name == tool_name)
    result = await selected_tool.ainvoke(payload)
    text = extract_tool_text(result)
    return {
        "server": server_spec.name,
        "tool_name": tool_name,
        "payload": payload,
        "raw_result": result,
        "text": text,
        "json_payload": parse_json_text(text),
    }


async def build_agent(
    server_spec: ServerSpec,
    *,
    openai_api_key: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    allowed_tool_names: Iterable[str] | None = None,
    extra_tools: Sequence[Any] | None = None,
    include_docling_tools: bool = False,
    response_format: type[BaseModel] | None = ResearchSubagentResult,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
):
    tools = list(
        await get_server_tools(
            server_spec,
            allowed_tool_names=allowed_tool_names,
            tool_interceptors=tool_interceptors,
        )
    )
    if include_docling_tools:
        from src.research.langchain_agent.tools_for_test.test_suite.docling_test_tools import (
            DOCLING_TEST_TOOLS,
        )

        tools.extend(DOCLING_TEST_TOOLS)
    if extra_tools:
        tools.extend(extra_tools)

    llm = ChatOpenAI(model=model_name, api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=server_spec.system_prompt,
        checkpointer=memory_saver,
        response_format=response_format,
    )
    return agent, tools


async def smoke_test_tool_discovery(
    server_spec: ServerSpec,
    *,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    """List tools and return a compact summary. No LLM call required."""
    tools = await get_server_tools(server_spec, tool_interceptors=tool_interceptors)
    return {
        "server": server_spec.name,
        "url": server_spec.url,
        "tool_count": len(tools),
        "tool_names": [tool.name for tool in tools],
    }


async def live_test_agent(
    server_spec: ServerSpec,
    *,
    prompt: str | None = None,
    openai_api_key: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    include_docling_tools: bool = False,
    response_format: type[BaseModel] | None = ResearchSubagentResult,
    tool_interceptors: Sequence[ToolCallInterceptor] | None = None,
) -> dict[str, Any]:
    """Run a real agent invocation. Requires network access and an OpenAI API key."""
    agent, tools = await build_agent(
        server_spec,
        openai_api_key=openai_api_key,
        model_name=model_name,
        include_docling_tools=include_docling_tools,
        response_format=response_format,
        tool_interceptors=tool_interceptors,
    )
    message = prompt or server_spec.default_test_prompt
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        },
        config={"configurable": {"thread_id": f"cyanheads-live-{server_spec.name}"}},
    )
    return {
        "server": server_spec.name,
        "tool_count": len(tools),
        "prompt": message,
        "final_text": extract_final_message_text(result),
        "structured_response": result.get("structured_response"),
        "result": result,
    }


async def main() -> None:
    interceptors = default_retry_interceptors()
    smoke_results = []
    capability_results = []
    for spec in SERVER_SPECS.values():
        try:
            smoke_results.append(
                await smoke_test_tool_discovery(spec, tool_interceptors=interceptors)
            )
            capability_results.append(
                await inspect_server_capabilities(spec, tool_interceptors=interceptors)
            )
        except Exception as exc:  # pragma: no cover - operational reporting
            smoke_results.append(
                {
                    "server": spec.name,
                    "url": spec.url,
                    "error": stringify_exception(exc),
                }
            )

    print("=== MCP TOOL DISCOVERY ===")
    print(json.dumps(smoke_results, indent=2, default=str))
    print("\n=== MCP CAPABILITIES ===")
    print(json.dumps(capability_results, indent=2, default=str))

    if os.getenv("RUN_LIVE_AGENT_TESTS", "0") == "1" and os.getenv("OPENAI_API_KEY"):
        print("\n=== LIVE AGENT RUNS ===")
        live_results = []
        for spec in SERVER_SPECS.values():
            try:
                live_results.append(
                    await live_test_agent(
                        spec,
                        model_name=DEFAULT_MODEL_NAME,
                        tool_interceptors=interceptors,
                    )
                )
            except Exception as exc:  # pragma: no cover - operational reporting
                live_results.append(
                    {
                        "server": spec.name,
                        "url": spec.url,
                        "error": stringify_exception(exc),
                    }
                )
        print(json.dumps(live_results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
