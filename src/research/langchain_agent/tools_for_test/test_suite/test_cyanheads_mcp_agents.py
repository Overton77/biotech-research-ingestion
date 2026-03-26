from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents import (
    DEFAULT_MODEL_NAME,
    MCPToolCallRequest,
    SERVER_SPECS,
    ResearchSubagentResult,
    RateLimitRetryInterceptor,
    RetryPolicy,
    assess_clinicaltrials_detail_payload,
    build_agent,
    fetch_server_prompt,
    fetch_server_resources,
    get_server_prompt_metadata,
    get_server_resource_metadata,
    get_server_tools,
    make_client,
    smoke_test_tool_discovery,
)


class FakeTool:
    def __init__(self, name: str, result=None):
        self.name = name
        self._result = result or [{"type": "text", "text": f"result for {name}"}]

    async def ainvoke(self, payload):
        return self._result


class FakePromptArgument:
    def __init__(self, name: str, description: str, required: bool):
        self.name = name
        self.description = description
        self.required = required


class FakePrompt:
    def __init__(self, name: str):
        self.name = name
        self.description = f"description for {name}"
        self.arguments = [
            FakePromptArgument("title", "Prompt title", True),
            FakePromptArgument("includeAgentPrompts", "Verbose guidance", False),
        ]


class FakeResource:
    def __init__(self, name: str, uri: str):
        self.name = name
        self.title = name
        self.uri = uri
        self.description = f"resource for {name}"
        self.mimeType = "application/json"


class FakeBlob:
    def __init__(self, text: str):
        self.mimetype = "application/json"
        self.source = "fake://resource"
        self._text = text

    def as_string(self) -> str:
        return self._text


class FakeSessionContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_prompts(self):
        return SimpleNamespace(prompts=[FakePrompt("research_plan")])

    async def list_resources(self):
        return SimpleNamespace(resources=[FakeResource("PubMed Database Info", "pubmed://database/info")])


class FakeMultiServerMCPClient:
    def __init__(self, config, callbacks=None, tool_interceptors=None, tool_name_prefix=False):
        self.config = config
        self.callbacks = callbacks
        self.tool_interceptors = tool_interceptors or []
        self.tool_name_prefix = tool_name_prefix

    async def get_tools(self, server_name=None):
        if server_name == "pubmed":
            return [
                FakeTool("pubmed_search_articles"),
                FakeTool("pubmed_fetch_articles"),
                FakeTool("pubmed_fetch_fulltext"),
                FakeTool("pubmed_format_citations"),
                FakeTool("pubmed_find_related"),
                FakeTool("pubmed_lookup_mesh"),
                FakeTool("pubmed_spell_check"),
                FakeTool("noise_tool"),
            ]
        if server_name == "pubchem":
            return [
                FakeTool("pubchem_search_compounds"),
                FakeTool("pubchem_get_compound_details"),
                FakeTool("pubchem_get_compound_safety"),
            ]
        if server_name == "clinicaltrials":
            return [
                FakeTool("clinicaltrials_search_studies"),
                FakeTool("clinicaltrials_get_study"),
                FakeTool("clinicaltrials_compare_studies"),
            ]
        return []

    def session(self, server_name, auto_initialize=True):
        return FakeSessionContext()

    async def get_prompt(self, server_name, prompt_name, arguments=None):
        return [
            SimpleNamespace(content="system guidance"),
            SimpleNamespace(content=f"prompt={prompt_name} args={arguments}"),
        ]

    async def get_resources(self, server_name=None, uris=None):
        return [FakeBlob('{"dbName":"pubmed","count":"123"}')]


@pytest.mark.asyncio
async def test_make_client_uses_headers_and_interceptors():
    interceptor = RateLimitRetryInterceptor()
    with (
        patch.dict("os.environ", {"PUBMED_MCP_BEARER_TOKEN": "secret-token"}, clear=False),
        patch(
            "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.MultiServerMCPClient",
            FakeMultiServerMCPClient,
        ),
    ):
        client = make_client(SERVER_SPECS["pubmed"], tool_interceptors=[interceptor])

    assert client.config["pubmed"]["transport"] == "http"
    assert client.config["pubmed"]["url"] == "https://pubmed.caseyjhand.com/mcp"
    assert client.config["pubmed"]["headers"]["Authorization"] == "Bearer secret-token"
    assert client.tool_interceptors == [interceptor]
    assert client.tool_name_prefix is False


@pytest.mark.asyncio
async def test_get_server_tools_filters_to_allowed_tools():
    with patch(
        "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    ):
        pubmed_tools = await get_server_tools(SERVER_SPECS["pubmed"])

    assert [tool.name for tool in pubmed_tools] == [
        "pubmed_search_articles",
        "pubmed_fetch_articles",
        "pubmed_fetch_fulltext",
        "pubmed_format_citations",
        "pubmed_find_related",
        "pubmed_lookup_mesh",
        "pubmed_spell_check",
    ]


@pytest.mark.asyncio
async def test_smoke_test_tool_discovery_returns_expected_summary():
    with patch(
        "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    ):
        clinical_summary = await smoke_test_tool_discovery(SERVER_SPECS["clinicaltrials"])

    assert clinical_summary["tool_names"] == [
        "clinicaltrials_search_studies",
        "clinicaltrials_get_study",
        "clinicaltrials_compare_studies",
    ]


@pytest.mark.asyncio
async def test_prompt_and_resource_metadata_are_serialized():
    with patch(
        "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    ):
        prompt_metadata = await get_server_prompt_metadata(SERVER_SPECS["pubmed"])
        resource_metadata = await get_server_resource_metadata(SERVER_SPECS["pubmed"])

    assert prompt_metadata["prompts"][0]["name"] == "research_plan"
    assert prompt_metadata["prompts"][0]["arguments"][0]["name"] == "title"
    assert resource_metadata["resources"][0]["uri"] == "pubmed://database/info"


@pytest.mark.asyncio
async def test_fetch_prompt_and_resources_capture_payloads():
    with patch(
        "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    ):
        prompt_payload = await fetch_server_prompt(
            SERVER_SPECS["pubmed"],
            "research_plan",
            arguments={"title": "Test"},
        )
        resource_payload = await fetch_server_resources(SERVER_SPECS["pubmed"])

    assert prompt_payload["message_count"] == 2
    assert "prompt=research_plan" in prompt_payload["messages"][1]["content"]
    assert resource_payload["resource_count"] == 1
    assert '"dbName":"pubmed"' in resource_payload["resources"][0]["text"]


def test_assess_clinicaltrials_detail_payload_marks_complete_interventional_record():
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test Sponsor"}},
            "descriptionModule": {"briefSummary": "summary"},
            "conditionsModule": {"conditions": ["Condition A"]},
            "eligibilityModule": {"minimumAge": "18 Years"},
            "designModule": {"studyType": "INTERVENTIONAL"},
            "armsInterventionsModule": {"interventions": [{"name": "Drug A"}]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "Response rate"}]},
        }
    }

    assessment = assess_clinicaltrials_detail_payload(payload)

    assert assessment["nct_id"] == "NCT01234567"
    assert assessment["direct_retrieval_sufficient"] is True
    assert assessment["missing_sections"] == []
    assert assessment["canonical_url"].endswith("/NCT01234567")


def test_assess_clinicaltrials_detail_payload_recommends_fallback_for_incomplete_record():
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT07654321"},
            "designModule": {"studyType": "INTERVENTIONAL"},
            "statusModule": {"overallStatus": "COMPLETED"},
        }
    }

    assessment = assess_clinicaltrials_detail_payload(payload)

    assert assessment["direct_retrieval_sufficient"] is False
    assert "sponsorCollaboratorsModule" in assessment["missing_sections"]
    assert assessment["fallback_actions"]


@pytest.mark.asyncio
async def test_retry_interceptor_retries_retryable_failures():
    attempts = {"count": 0}
    interceptor = RateLimitRetryInterceptor(
        RetryPolicy(max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0, jitter_seconds=0.0)
    )
    request = MCPToolCallRequest(
        name="clinicaltrials_get_study",
        args={"nctIds": "NCT01234567"},
        server_name="clinicaltrials",
    )

    async def flaky_handler(_request):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("429 Too Many Requests")
        return {"ok": True}

    result = await interceptor(request, flaky_handler)

    assert result == {"ok": True}
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_interceptor_does_not_retry_validation_errors():
    attempts = {"count": 0}
    interceptor = RateLimitRetryInterceptor(
        RetryPolicy(max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0, jitter_seconds=0.0)
    )
    request = MCPToolCallRequest(
        name="pubmed_fetch_articles",
        args={"pmids": ["bad"]},
        server_name="pubmed",
    )

    async def invalid_handler(_request):
        attempts["count"] += 1
        raise ValueError("Input validation error")

    with pytest.raises(ValueError):
        await interceptor(request, invalid_handler)

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_build_agent_passes_model_tools_and_response_format():
    captured = {}

    async def fake_get_server_tools(*args, **kwargs):
        return [FakeTool("pubmed_search_articles")]

    class FakeChatOpenAI:
        def __init__(self, model, api_key):
            captured["model_name"] = model
            captured["api_key"] = api_key

    def fake_create_agent(**kwargs):
        captured["create_agent_kwargs"] = kwargs
        return "fake-agent"

    with (
        patch(
            "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.get_server_tools",
            fake_get_server_tools,
        ),
        patch(
            "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.ChatOpenAI",
            FakeChatOpenAI,
        ),
        patch(
            "src.research.langchain_agent.tools_for_test.test_suite.cyanheads_mcp_agents.create_agent",
            fake_create_agent,
        ),
    ):
        agent, tools = await build_agent(
            SERVER_SPECS["pubmed"],
            openai_api_key="test-key",
            model_name=DEFAULT_MODEL_NAME,
            response_format=ResearchSubagentResult,
        )

    assert agent == "fake-agent"
    assert [tool.name for tool in tools] == ["pubmed_search_articles"]
    assert captured["model_name"] == DEFAULT_MODEL_NAME
    assert captured["api_key"] == "test-key"
    assert captured["create_agent_kwargs"]["response_format"] is ResearchSubagentResult
    assert captured["create_agent_kwargs"]["system_prompt"] == SERVER_SPECS["pubmed"].system_prompt
