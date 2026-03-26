from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import aiofiles
import aiohttp
from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent

from src.research.langchain_agent.utils import save_json_artifact, save_text_artifact

load_dotenv()

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.35
DEFAULT_MAX_RETRIES = 5
DEFAULT_RUN_NAME = "clinical_trials_v2"
_MODULE_DIR = Path(__file__).resolve().parent

DEFAULT_STUDY_FIELDS: tuple[str, ...] = (
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.armsInterventionsModule.interventions",
)

SEARCH_GUIDE_AREA_NAMES: dict[str, str] = {
    "basic": "BasicSearch",
    "condition": "ConditionSearch",
    "intervention": "InterventionSearch",
    "sponsor": "SponsorSearch",
}

MESH_FIELD_BY_FOCUS: dict[str, str] = {
    "condition": "ConditionMeshTerm",
    "intervention": "InterventionMeshTerm",
}

_CLINICAL_TRIALS_AGENT_PROMPT = """
You are a ClinicalTrials.gov v2 research agent.

Workflow guidance:
- For sponsor-company searches, prefer `clinicaltrials_search_studies_tool` with `lead_sponsor`,
  then `sponsor` if collaborator coverage is also needed.
- For condition or intervention searches, prefer the dedicated `condition` or `intervention`
  parameters before reaching for `advanced_query`.
- Use `clinicaltrials_query_syntax_guide` before composing advanced `AREA[...]` queries or
  MeSH-oriented searches.
- When an NCT ID is known, use `clinicaltrials_get_study_tool` or
  `clinicaltrials_download_study_tool` instead of searching again.
- Preserve exact NCT IDs, sponsor names, recruitment status, interventions, canonical study
  URLs, and any written artifact paths.
- Avoid broad fishing when a sponsor, condition, intervention, or identifier is already available.
""".strip()


class ClinicalTrialsGovError(RuntimeError):
    """Raised when the ClinicalTrials.gov API returns a terminal error."""


@dataclass(slots=True)
class _ApiResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ClinicalTrialsGovError(
                f"{self.status_code} error for {self.url}: {self.text[:400]}"
            )


def _tool_command(runtime: ToolRuntime, content: str) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id),
            ]
        }
    )


def _format_tool_error(
    *,
    tool_name: str,
    error: Exception,
    details: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"ClinicalTrials.gov tool error: `{tool_name}`",
        f"Type: {type(error).__name__}",
        f"Message: {error}",
    ]

    if details:
        lines.append("")
        lines.append("Inputs:")
        for key, value in details.items():
            if value is None or value == "" or value is False:
                continue
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("What to try next:")
    lines.append("- Verify the query shape or identifier and retry with a narrower request.")
    lines.append("- Use `clinicaltrials_query_syntax_guide` if you are composing an AREA[...] or sponsor query.")
    lines.append("- If this was a download request, try fetching the study detail first by NCT ID.")
    return "\n".join(lines)


def _tool_error_command(
    runtime: ToolRuntime,
    *,
    tool_name: str,
    error: Exception,
    details: dict[str, Any] | None = None,
) -> Command:
    return _tool_command(
        runtime,
        _format_tool_error(tool_name=tool_name, error=error, details=details),
    )


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    max_attempts: int = DEFAULT_MAX_RETRIES
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 12.0
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


class RequestThrottle:
    """Serialize calls enough to avoid hammering the public API."""

    def __init__(self, min_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def wait_turn(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_allowed_at:
                await asyncio.sleep(self._next_allowed_at - now)
                now = loop.time()
            self._next_allowed_at = now + self.min_interval_seconds


class ClinicalTrialsGovClient:
    def __init__(
        self,
        *,
        base_url: str = CTGOV_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry_policy: RetryPolicy | None = None,
        throttle: RequestThrottle | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.throttle = throttle or RequestThrottle()
        self._client: aiohttp.ClientSession | None = None

    async def _ensure_client(self) -> aiohttp.ClientSession:
        if self._client is None or self._client.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
            self._client = aiohttp.ClientSession(
                timeout=timeout,
                raise_for_status=False,
                trust_env=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                connector=connector,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.closed:
            await self._client.close()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _ApiResponse:
        client = await self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            await self.throttle.wait_turn()
            try:
                merged_headers = dict(client.headers)
                if headers:
                    merged_headers.update(headers)
                url = f"{self.base_url}{path}"

                async with client.request(
                    method,
                    url,
                    params=params,
                    headers=merged_headers,
                ) as raw_response:
                    body = await raw_response.read()
                    response = _ApiResponse(
                        status_code=raw_response.status,
                        headers=dict(raw_response.headers),
                        content=body,
                        url=str(raw_response.url),
                    )

                if response.status_code in self.retry_policy.retryable_status_codes:
                    if attempt >= self.retry_policy.max_attempts:
                        response.raise_for_status()
                    await asyncio.sleep(self._retry_delay_seconds(response, attempt))
                    continue
                response.raise_for_status()
                return response
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt >= self.retry_policy.max_attempts:
                    break
                await asyncio.sleep(self._retry_delay_seconds(None, attempt))

        raise ClinicalTrialsGovError(f"Request failed for {path}: {last_error}") from last_error

    def _retry_delay_seconds(
        self,
        response: _ApiResponse | None,
        attempt: int,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.5, float(retry_after))
                except ValueError:
                    pass

        base_delay = min(
            self.retry_policy.max_delay_seconds,
            self.retry_policy.base_delay_seconds * (2 ** (attempt - 1)),
        )
        return base_delay + random.uniform(0.0, 0.35)

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = await self._request("GET", path, params=params)
        return response.json()

    async def get_bytes(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> bytes:
        headers = {"Accept": accept} if accept else None
        response = await self._request("GET", path, params=params, headers=headers)
        return response.content

    async def get_version(self) -> dict[str, Any]:
        return await self.get_json("/version")

    async def get_studies_metadata(self) -> list[dict[str, Any]]:
        return await self.get_json("/studies/metadata")

    async def get_search_areas(self) -> list[dict[str, Any]]:
        return await self.get_json("/studies/search-areas")

    async def search_studies(
        self,
        *,
        query: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        sponsor: str | None = None,
        lead_sponsor: str | None = None,
        study_id: str | None = None,
        advanced_query: str | None = None,
        fields: Sequence[str] | None = None,
        page_size: int = 10,
        next_page_token: str | None = None,
        sort: str | None = None,
        count_total: bool = False,
        response_format: Literal["json", "csv"] = "json",
    ) -> dict[str, Any]:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")

        params: dict[str, Any] = {
            "format": response_format,
            "pageSize": page_size,
        }
        if advanced_query:
            params["query.term"] = advanced_query
        elif query:
            params["query.term"] = query
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if sponsor:
            params["query.spons"] = sponsor
        if lead_sponsor:
            params["query.lead"] = lead_sponsor
        if study_id:
            params["query.id"] = study_id
        if fields:
            params["fields"] = ",".join(fields)
        if next_page_token:
            params["pageToken"] = next_page_token
        if sort:
            params["sort"] = sort
        if count_total:
            params["countTotal"] = "true"

        return await self.get_json("/studies", params=params)

    async def collect_studies(
        self,
        *,
        max_pages: int = 1,
        **search_kwargs: Any,
    ) -> dict[str, Any]:
        studies: list[dict[str, Any]] = []
        next_page_token: str | None = None
        first_payload: dict[str, Any] | None = None

        for page_index in range(max_pages):
            payload = await self.search_studies(
                next_page_token=next_page_token,
                **search_kwargs,
            )
            if first_payload is None:
                first_payload = payload
            studies.extend(payload.get("studies") or [])
            next_page_token = payload.get("nextPageToken")
            if not next_page_token or page_index + 1 >= max_pages:
                break

        return {
            "studies": studies,
            "pagesFetched": page_index + 1 if first_payload is not None else 0,
            "nextPageToken": next_page_token,
            "totalCount": (first_payload or {}).get("totalCount"),
        }

    async def get_study(
        self,
        nct_id: str,
        *,
        fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        return await self.get_json(f"/studies/{nct_id}", params=params or None)

    async def download_study_json(
        self,
        nct_id: str,
        destination: Path,
        *,
        fields: Sequence[str] | None = None,
    ) -> Path:
        payload = await self.get_study(nct_id, fields=fields)
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(destination, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return destination

    async def download_bulk_archive(
        self,
        destination: Path,
        *,
        archive_format: str = "json.zip",
    ) -> Path:
        content = await self.get_bytes(
            "/studies/download",
            params={"format": archive_format},
            accept="application/zip",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(content)
        return destination


_shared_client: ClinicalTrialsGovClient | None = None
_shared_client_lock = asyncio.Lock()


async def get_clinical_trials_client() -> ClinicalTrialsGovClient:
    global _shared_client
    if _shared_client is not None:
        return _shared_client

    async with _shared_client_lock:
        if _shared_client is None:
            _shared_client = ClinicalTrialsGovClient()
        return _shared_client


async def close_clinical_trials_client() -> None:
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


def _flatten_metadata_items(items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}

    def visit(nodes: Sequence[dict[str, Any]]) -> None:
        for node in nodes:
            piece = node.get("piece")
            if piece:
                flattened[piece] = node
            children = node.get("children") or []
            if children:
                visit(children)

    visit(items)
    return flattened


def _find_area(search_areas: Sequence[dict[str, Any]], area_name: str) -> dict[str, Any] | None:
    for section in search_areas:
        for area in section.get("areas") or []:
            if area.get("name") == area_name:
                return area
    return None


def _study_record(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("study") or payload


def _extract_summary(study: dict[str, Any]) -> dict[str, Any]:
    protocol = study.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    sponsor_module = protocol.get("sponsorCollaboratorsModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    interventions_module = protocol.get("armsInterventionsModule") or {}

    interventions = interventions_module.get("interventions") or []
    intervention_names = [
        intervention.get("name")
        for intervention in interventions
        if isinstance(intervention, dict) and intervention.get("name")
    ]

    lead_sponsor = (sponsor_module.get("leadSponsor") or {}).get("name")
    collaborators = sponsor_module.get("collaborators") or []
    collaborator_names = [
        collaborator.get("name")
        for collaborator in collaborators
        if isinstance(collaborator, dict) and collaborator.get("name")
    ]

    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "overall_status": status.get("overallStatus"),
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "conditions": conditions_module.get("conditions") or [],
        "lead_sponsor": lead_sponsor,
        "collaborators": collaborator_names,
        "interventions": intervention_names,
    }


def _format_study_lines(study: dict[str, Any], index: int | None = None) -> list[str]:
    summary = _extract_summary(study)
    prefix = f"[{index}] " if index is not None else ""
    lines = [
        f"{prefix}{summary['nct_id'] or 'Unknown NCT'} | {summary['brief_title'] or 'Untitled study'}",
        f"Status: {summary['overall_status'] or 'unknown'}",
    ]
    if summary["lead_sponsor"]:
        lines.append(f"Lead sponsor: {summary['lead_sponsor']}")
    if summary["conditions"]:
        lines.append(f"Conditions: {', '.join(summary['conditions'][:4])}")
    if summary["interventions"]:
        lines.append(f"Interventions: {', '.join(summary['interventions'][:4])}")
    if summary["collaborators"]:
        lines.append(f"Collaborators: {', '.join(summary['collaborators'][:4])}")
    lines.append(f"Study URL: https://clinicaltrials.gov/study/{summary['nct_id']}")
    return lines


def format_search_payload(payload: dict[str, Any]) -> str:
    studies = payload.get("studies") or []
    lines = [
        "=== ClinicalTrials.gov Search Results ===",
        f"Returned studies: {len(studies)}",
    ]
    if payload.get("totalCount") is not None:
        lines.append(f"Total count: {payload['totalCount']}")
    lines.append(f"Pages fetched: {payload.get('pagesFetched', 1)}")

    if not studies:
        lines.append("No studies matched the query.")
        return "\n".join(lines)

    for index, study in enumerate(studies[:10], start=1):
        lines.append("")
        lines.extend(_format_study_lines(study, index=index))

    if payload.get("nextPageToken"):
        lines.append("")
        lines.append("More studies are available via nextPageToken.")

    return "\n".join(lines)


def format_study_payload(payload: dict[str, Any]) -> str:
    study = _study_record(payload)
    lines = ["=== ClinicalTrials.gov Study Detail ===", ""]
    lines.extend(_format_study_lines(study))

    protocol = study.get("protocolSection") or {}
    description_module = protocol.get("descriptionModule") or {}
    summary = description_module.get("briefSummary")
    detailed = description_module.get("detailedDescription")
    if summary:
        lines.append("")
        lines.append("Brief summary:")
        lines.append(summary[:1200])
    if detailed:
        lines.append("")
        lines.append("Detailed description:")
        lines.append(detailed[:1600])
    return "\n".join(lines)


async def build_query_syntax_guide(
    *,
    focus: Literal["basic", "condition", "intervention", "sponsor"] = "condition",
    sample_term: str | None = None,
    include_live_examples: bool = True,
) -> str:
    client = await get_clinical_trials_client()
    search_areas = await client.get_search_areas()
    metadata = await client.get_studies_metadata()
    flattened_metadata = _flatten_metadata_items(metadata)

    area_name = SEARCH_GUIDE_AREA_NAMES[focus]
    area = _find_area(search_areas, area_name)
    if area is None:
        raise ClinicalTrialsGovError(f"Search area {area_name} not found.")

    lines = [
        "=== ClinicalTrials.gov Query Guide ===",
        f"Focus: {focus}",
        f"API parameter: query.{area.get('param', 'term')}",
    ]
    if area.get("uiLabel"):
        lines.append(f"UI label: {area['uiLabel']}")

    ranked_parts = sorted(
        area.get("parts") or [],
        key=lambda item: item.get("weight", 0.0),
        reverse=True,
    )
    lines.append("")
    lines.append("Top weighted fields in this search area:")
    for part in ranked_parts[:8]:
        pieces = ", ".join(part.get("pieces") or [])
        lines.append(
            f"- {pieces} | weight={part.get('weight')} | synonyms={part.get('isSynonyms', False)}"
        )

    sample_term = sample_term or {
        "basic": "melanoma pembrolizumab",
        "condition": "melanoma",
        "intervention": "pembrolizumab",
        "sponsor": "Pfizer",
    }[focus]

    lines.append("")
    lines.append("Recommended query patterns:")
    if focus == "condition":
        mesh_piece = flattened_metadata.get("ConditionMeshTerm")
        if mesh_piece and mesh_piece.get("description"):
            lines.append(f"- ConditionMeshTerm: {mesh_piece['description']}")
        lines.append(f"- query.cond={sample_term}")
        lines.append(f"- query.term=AREA[ConditionMeshTerm]{sample_term}")
        lines.append(
            f"- query.term=AREA[ConditionSearch]{sample_term} AND AREA[LeadSponsorName]Pfizer"
        )
    elif focus == "intervention":
        mesh_piece = flattened_metadata.get("InterventionMeshTerm")
        if mesh_piece and mesh_piece.get("description"):
            lines.append(f"- InterventionMeshTerm: {mesh_piece['description']}")
        lines.append(f"- query.intr={sample_term}")
        lines.append(f"- query.term=AREA[InterventionMeshTerm]{sample_term}")
        lines.append(
            f"- query.term=AREA[InterventionName]{sample_term} AND AREA[ConditionSearch]melanoma"
        )
    elif focus == "sponsor":
        lead_piece = flattened_metadata.get("LeadSponsorName")
        if lead_piece and lead_piece.get("description"):
            lines.append(f"- LeadSponsorName: {lead_piece['description']}")
        lines.append(f"- query.lead={sample_term}")
        lines.append(f"- query.spons={sample_term}")
        lines.append(
            f"- query.term=AREA[LeadSponsorName]{sample_term} AND AREA[StudyType]INTERVENTIONAL"
        )
    else:
        lines.append(f"- query.term={sample_term}")
        lines.append(
            f"- query.term=AREA[ConditionSearch]melanoma AND AREA[InterventionName]pembrolizumab"
        )
        lines.append(
            f"- query.term=AREA[LeadSponsorName]Pfizer AND AREA[ConditionMeshTerm]melanoma"
        )

    if include_live_examples:
        lines.append("")
        lines.append("Live example matches:")
        if focus == "condition":
            live = await client.collect_studies(
                advanced_query=f"AREA[ConditionMeshTerm]{sample_term}",
                fields=DEFAULT_STUDY_FIELDS,
                page_size=3,
            )
        elif focus == "intervention":
            live = await client.collect_studies(
                advanced_query=f"AREA[InterventionMeshTerm]{sample_term}",
                fields=DEFAULT_STUDY_FIELDS,
                page_size=3,
            )
        elif focus == "sponsor":
            live = await client.collect_studies(
                lead_sponsor=sample_term,
                fields=DEFAULT_STUDY_FIELDS,
                page_size=3,
            )
        else:
            live = await client.collect_studies(
                query=sample_term,
                fields=DEFAULT_STUDY_FIELDS,
                page_size=3,
            )
        studies = live.get("studies") or []
        if not studies:
            lines.append("- No live example studies were returned for that sample term.")
        else:
            for study in studies[:3]:
                summary = _extract_summary(study)
                lines.append(
                    f"- {summary['nct_id']} | {summary['brief_title']} | sponsor={summary['lead_sponsor'] or 'unknown'}"
                )

    return "\n".join(lines)


async def save_search_payload(
    payload: dict[str, Any],
    *,
    label: str,
    run_name: str = DEFAULT_RUN_NAME,
) -> dict[str, Path]:
    formatted = format_search_payload(payload)
    suffix = label[:72]
    json_path = await save_json_artifact(
        payload,
        run_name,
        "clinical_trials_search_raw",
        suffix=suffix,
        base_dir=_MODULE_DIR,
    )
    text_path = await save_text_artifact(
        formatted,
        run_name,
        "clinical_trials_search_formatted",
        suffix=suffix,
        base_dir=_MODULE_DIR,
        extension="md",
    )
    return {"json": json_path, "text": text_path}


async def save_study_payload(
    payload: dict[str, Any],
    *,
    nct_id: str,
    run_name: str = DEFAULT_RUN_NAME,
) -> dict[str, Path]:
    formatted = format_study_payload(payload)
    json_path = await save_json_artifact(
        payload,
        run_name,
        "clinical_trial_detail",
        suffix=nct_id,
        base_dir=_MODULE_DIR,
    )
    text_path = await save_text_artifact(
        formatted,
        run_name,
        "clinical_trial_detail",
        suffix=nct_id,
        base_dir=_MODULE_DIR,
        extension="md",
    )
    return {"json": json_path, "text": text_path}


@tool(
    description=(
        "Explain ClinicalTrials.gov v2 query syntax, search areas, and MeSH-related patterns. "
        "Use this before composing advanced AREA[...] expressions or MeSH-oriented searches."
    ),
    parse_docstring=False,
)
async def clinicaltrials_query_syntax_guide(
    runtime: ToolRuntime,
    focus: Annotated[
        Literal["basic", "condition", "intervention", "sponsor"],
        "Which search mode to explain.",
    ] = "condition",
    sample_term: Annotated[
        str | None,
        "Optional example term to illustrate AREA[...] or query.* usage.",
    ] = None,
    include_live_examples: Annotated[
        bool,
        "Whether to run a small live example query against the API.",
    ] = True,
) -> Command:
    try:
        guide = await build_query_syntax_guide(
            focus=focus,
            sample_term=sample_term,
            include_live_examples=include_live_examples,
        )
        return _tool_command(runtime, guide)
    except Exception as error:
        return _tool_error_command(
            runtime,
            tool_name="clinicaltrials_query_syntax_guide",
            error=error,
            details={
                "focus": focus,
                "sample_term": sample_term,
                "include_live_examples": include_live_examples,
            },
        )


@tool(
    description=(
        "Search ClinicalTrials.gov studies by sponsor company, lead sponsor, condition, intervention, "
        "study identifier, free-text query, or advanced AREA[...] query syntax."
    ),
    parse_docstring=False,
)
async def clinicaltrials_search_studies_tool(
    runtime: ToolRuntime,
    query: Annotated[str | None, "Broad free-text query mapped to query.term."] = None,
    condition: Annotated[str | None, "Condition-specific query mapped to query.cond."] = None,
    intervention: Annotated[str | None, "Intervention-specific query mapped to query.intr."] = None,
    sponsor: Annotated[str | None, "Broad sponsor/collaborator search mapped to query.spons."] = None,
    lead_sponsor: Annotated[str | None, "Lead sponsor company search mapped to query.lead."] = None,
    study_id: Annotated[str | None, "NCT ID or alternate study identifier mapped to query.id."] = None,
    advanced_query: Annotated[
        str | None,
        "Advanced AREA[...] boolean query mapped to query.term.",
    ] = None,
    page_size: Annotated[int, "Studies per page (1-1000)."] = 10,
    max_pages: Annotated[int, "How many pages to fetch sequentially."] = 1,
    save_artifacts: Annotated[bool, "Whether to save formatted and raw search artifacts to disk."] = False,
) -> Command:
    try:
        client = await get_clinical_trials_client()
        payload = await client.collect_studies(
            query=query,
            condition=condition,
            intervention=intervention,
            sponsor=sponsor,
            lead_sponsor=lead_sponsor,
            study_id=study_id,
            advanced_query=advanced_query,
            fields=DEFAULT_STUDY_FIELDS,
            page_size=page_size,
            max_pages=max_pages,
            count_total=True,
        )
        formatted = format_search_payload(payload)

        if save_artifacts:
            label = lead_sponsor or sponsor or condition or intervention or study_id or advanced_query or query or "search"
            paths = await save_search_payload(payload, label=label)
            formatted += (
                "\n\nArtifacts:\n"
                f"- raw: {paths['json']}\n"
                f"- summary: {paths['text']}"
            )

        return _tool_command(runtime, formatted)
    except Exception as error:
        return _tool_error_command(
            runtime,
            tool_name="clinicaltrials_search_studies_tool",
            error=error,
            details={
                "query": query,
                "condition": condition,
                "intervention": intervention,
                "sponsor": sponsor,
                "lead_sponsor": lead_sponsor,
                "study_id": study_id,
                "advanced_query": advanced_query,
                "page_size": page_size,
                "max_pages": max_pages,
                "save_artifacts": save_artifacts,
            },
        )


@tool(
    description=(
        "Fetch one ClinicalTrials.gov study by NCT identifier and optionally save the full JSON payload to disk."
    ),
    parse_docstring=False,
)
async def clinicaltrials_get_study_tool(
    runtime: ToolRuntime,
    nct_id: Annotated[str, "The NCT identifier, for example NCT04267848."],
    save_artifacts: Annotated[bool, "Whether to save the full study payload to disk."] = True,
) -> Command:
    try:
        client = await get_clinical_trials_client()
        payload = await client.get_study(nct_id)
        formatted = format_study_payload(payload)

        if save_artifacts:
            paths = await save_study_payload(payload, nct_id=nct_id)
            formatted += (
                "\n\nArtifacts:\n"
                f"- raw: {paths['json']}\n"
                f"- summary: {paths['text']}"
            )

        return _tool_command(runtime, formatted)
    except Exception as error:
        return _tool_error_command(
            runtime,
            tool_name="clinicaltrials_get_study_tool",
            error=error,
            details={
                "nct_id": nct_id,
                "save_artifacts": save_artifacts,
            },
        )


@tool(
    description=(
        "Download a full ClinicalTrials.gov study JSON payload to the filesystem for a known NCT ID."
    ),
    parse_docstring=False,
)
async def clinicaltrials_download_study_tool(
    runtime: ToolRuntime,
    nct_id: Annotated[str, "The NCT identifier to download, for example NCT04267848."],
    filename: Annotated[
        str | None,
        "Optional output filename. Defaults to <NCT>.json under the module artifacts directory.",
    ] = None,
) -> Command:
    try:
        client = await get_clinical_trials_client()
        target = (
            _MODULE_DIR
            / "artifacts"
            / DEFAULT_RUN_NAME
            / (filename or f"{nct_id}.json")
        )
        path = await client.download_study_json(nct_id, target)
        content = f"Downloaded full study payload for {nct_id} to {path}"
        return _tool_command(runtime, content)
    except Exception as error:
        return _tool_error_command(
            runtime,
            tool_name="clinicaltrials_download_study_tool",
            error=error,
            details={
                "nct_id": nct_id,
                "filename": filename,
            },
        )


@tool(
    description=(
        "Download the ClinicalTrials.gov bulk archive as a json.zip file. Use this for full-registry snapshots, "
        "not for single-trial lookup."
    ),
    parse_docstring=False,
)
async def clinicaltrials_download_bulk_archive_tool(
    runtime: ToolRuntime,
    filename: Annotated[
        str,
        "Destination filename for the bulk archive ZIP.",
    ] = "ctgov_studies.json.zip",
) -> Command:
    try:
        client = await get_clinical_trials_client()
        destination = _MODULE_DIR / "artifacts" / DEFAULT_RUN_NAME / filename
        path = await client.download_bulk_archive(destination)
        content = f"Downloaded ClinicalTrials.gov bulk archive to {path}"
        return _tool_command(runtime, content)
    except Exception as error:
        return _tool_error_command(
            runtime,
            tool_name="clinicaltrials_download_bulk_archive_tool",
            error=error,
            details={
                "filename": filename,
            },
        )


clinical_trials_test_tools = [
    clinicaltrials_query_syntax_guide,
    clinicaltrials_search_studies_tool,
    clinicaltrials_get_study_tool,
    clinicaltrials_download_study_tool,
    clinicaltrials_download_bulk_archive_tool,
]


def create_clinical_trials_agent(*, model: str = "gpt-5.4-mini"):
    return create_agent(
        model=model,
        tools=clinical_trials_test_tools,
        system_prompt=_CLINICAL_TRIALS_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )


__all__ = [
    "ClinicalTrialsGovClient",
    "ClinicalTrialsGovError",
    "clinical_trials_test_tools",
    "clinicaltrials_download_bulk_archive_tool",
    "clinicaltrials_download_study_tool",
    "clinicaltrials_get_study_tool",
    "clinicaltrials_query_syntax_guide",
    "clinicaltrials_search_studies_tool",
    "close_clinical_trials_client",
    "create_clinical_trials_agent",
    "get_clinical_trials_client",
]