from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from src.research.langchain_agent.agent.config import ROOT_FILESYSTEM, MissionSliceInput
from src.research.langchain_agent.unstructured.candidate_vetting import (
    vet_stage_candidates,
)
from src.research.langchain_agent.unstructured.models import (
    CandidateDocument,
    CandidateProvenance,
    MissionCandidateManifest,
    StageCandidateManifest,
)
from src.research.langchain_agent.unstructured.paths import (
    mission_candidate_dir,
    relative_to_root,
    stage_candidate_dir,
)

DOCUMENT_SUFFIXES = {
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _candidate_id(seed: str) -> str:
    return _sha1_text(seed)[:20]


def _dedupe_key(*parts: str) -> str:
    normalized = "||".join((part or "").strip().lower() for part in parts if part)
    return _sha1_text(normalized) if normalized else _sha1_text("empty")


def _is_document_path(path: str) -> bool:
    return Path(path).suffix.lower() in DOCUMENT_SUFFIXES


def _guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or ""


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_sandbox_path(root: Path, relative_path: str) -> Path:
    direct = root / relative_path
    if direct.exists():
        return direct
    workspace = root / "workspace" / relative_path
    if workspace.exists():
        return workspace
    repo_root = root.parent.parent.parent.parent
    return repo_root / relative_path


def _iter_subagent_handoffs(task_slug: str, *, root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    handoffs: list[tuple[str, dict[str, Any]]] = []
    for subagent_root in [
        root / "runs" / task_slug / "subagents",
        root / "workspace" / "runs" / task_slug / "subagents",
        root.parent.parent.parent.parent / "runs" / task_slug / "subagents",
    ]:
        if not subagent_root.exists():
            continue
        for handoff_path in sorted(subagent_root.glob("*/handoff.json")):
            payload = _load_json_if_exists(handoff_path)
            if payload is None:
                continue
            relative_handoff = str(handoff_path).replace(str(root / "workspace") + "\\", "").replace(str(root / "workspace") + "/", "")
            relative_handoff = str(relative_handoff).replace(str(root) + "\\", "").replace(str(root) + "/", "")
            handoffs.append((relative_handoff.replace("\\", "/"), payload))
    return handoffs


def _candidate_from_local_path(
    *,
    run_input: MissionSliceInput,
    local_abs: Path,
    relative_path: str,
    discovered_by: str,
    source_artifact_path: str,
    reasons: list[str],
    metadata: dict[str, Any] | None = None,
    source_type: str = "local_file",
) -> CandidateDocument | None:
    if not local_abs.exists() or not local_abs.is_file():
        return None

    title = local_abs.stem
    mime_type = _guess_mime_type(local_abs.name)
    provenance = CandidateProvenance(
        mission_id=run_input.mission_id,
        task_slug=run_input.task_slug,
        discovered_by=discovered_by,
        source_artifact_path=source_artifact_path,
        notes=list(reasons),
    )
    dedupe_key = _dedupe_key(relative_path, metadata.get("accession_number", "") if metadata else "")
    return CandidateDocument(
        candidate_id=_candidate_id(f"{run_input.task_slug}:{relative_path}"),
        dedupe_key=dedupe_key,
        source_type=source_type,  # type: ignore[arg-type]
        title=title,
        mime_type=mime_type,
        local_path=str(local_abs),
        relative_path=relative_path,
        issuer_name=(metadata or {}).get("company", "") or (metadata or {}).get("issuer_name", ""),
        issuer_ticker=(metadata or {}).get("ticker", "") or (metadata or {}).get("issuer_ticker", ""),
        form_type=(metadata or {}).get("form", "") or (metadata or {}).get("form_type", ""),
        accession_number=(metadata or {}).get("accession_no", "") or (metadata or {}).get("accession_number", ""),
        filing_date=(metadata or {}).get("filing_date", ""),
        priority="low",
        review_status="provisional",
        reasons=reasons,
        metadata=metadata or {},
        provenance=provenance,
    )


def _candidate_from_url(
    *,
    run_input: MissionSliceInput,
    url: str,
    discovered_by: str,
    reason: str,
) -> CandidateDocument | None:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES and "sec.gov/archives" not in url.lower():
        return None

    provenance = CandidateProvenance(
        mission_id=run_input.mission_id,
        task_slug=run_input.task_slug,
        discovered_by=discovered_by,
        source_url=url,
        notes=[reason],
    )
    title = Path(parsed.path).name or parsed.netloc
    return CandidateDocument(
        candidate_id=_candidate_id(f"{run_input.task_slug}:{url}"),
        dedupe_key=_dedupe_key(url),
        source_type="url",
        title=title,
        uri=url,
        mime_type=_guess_mime_type(title),
        priority="low",
        review_status="provisional",
        reasons=[reason],
        metadata={"hostname": parsed.netloc},
        provenance=provenance,
    )


async def build_stage_candidate_manifest(
    *,
    run_input: MissionSliceInput,
    result: dict[str, Any],
    root: Path = ROOT_FILESYSTEM,
) -> tuple[StageCandidateManifest, str]:
    candidate_dir = stage_candidate_dir(run_input.task_slug, root=root)
    candidates_by_key: dict[str, CandidateDocument] = {}
    skipped_items: list[dict[str, Any]] = []

    def register(candidate: CandidateDocument | None) -> None:
        if candidate is None:
            return
        existing = candidates_by_key.get(candidate.dedupe_key)
        if existing is None:
            candidates_by_key[candidate.dedupe_key] = candidate
            return
        existing.reasons = sorted(set(existing.reasons + candidate.reasons))
        existing.metadata.update(candidate.metadata)
        if not existing.local_path and candidate.local_path:
            existing.local_path = candidate.local_path
            existing.relative_path = candidate.relative_path
        if not existing.uri and candidate.uri:
            existing.uri = candidate.uri

    for rel_path in result.get("written_file_paths", []) or []:
        if not _is_document_path(rel_path):
            continue
        local_abs = _resolve_sandbox_path(root, rel_path)
        register(
            _candidate_from_local_path(
                run_input=run_input,
                local_abs=local_abs,
                relative_path=rel_path,
                discovered_by="stage_written_file",
                source_artifact_path=rel_path,
                reasons=["Agent wrote a document-like artifact during the stage."],
            )
        )

    for url in result.get("visited_urls", []) or []:
        register(
            _candidate_from_url(
                run_input=run_input,
                url=url,
                discovered_by="visited_url",
                reason="Visited a URL that looks like a document or SEC archive artifact.",
            )
        )

    for handoff_rel_path, handoff in _iter_subagent_handoffs(run_input.task_slug, root=root):
        subagent_name = handoff.get("subagent_name", "")
        for artifact in handoff.get("artifacts", []) or []:
            artifact_path = str(artifact.get("path", "")).replace("\\", "/")
            if not artifact_path or not _is_document_path(artifact_path):
                continue
            local_abs = _resolve_sandbox_path(root, artifact_path)
            metadata: dict[str, Any] = {}
            artifact_lower = artifact_path.lower()
            if "edgar" in subagent_name.lower() or "edgar" in artifact_lower:
                metadata["source_family"] = "edgar"
            register(
                _candidate_from_local_path(
                    run_input=run_input,
                    local_abs=local_abs,
                    relative_path=artifact_path,
                    discovered_by=f"subagent:{subagent_name}",
                    source_artifact_path=handoff_rel_path,
                    reasons=[f"Subagent `{subagent_name}` produced a document artifact for downstream ingestion."],
                    metadata=metadata,
                    source_type="report_artifact",
                )
            )

    # Recognize downloaded Edgar filings by metadata files.
    for filings_root in [
        candidate_dir.parent / "edgar",
        root / "runs" / run_input.task_slug / "subagents" / "edgar_research",
        root / "runs" / run_input.task_slug / "unstructured" / "edgar",
        root / "workspace" / "runs" / run_input.task_slug / "unstructured" / "edgar",
        root / "workspace" / "runs" / run_input.task_slug / "subagents" / "edgar_research",
        root.parent.parent.parent.parent / "runs" / run_input.task_slug / "unstructured" / "edgar",
        root.parent.parent.parent.parent / "runs" / run_input.task_slug / "subagents" / "edgar_research",
    ]:
        if not filings_root.exists():
            continue
        for metadata_path in sorted(filings_root.glob("**/metadata.json")):
            metadata = _load_json_if_exists(metadata_path)
            if metadata is None:
                skipped_items.append({"path": str(metadata_path), "reason": "Invalid metadata JSON"})
                continue

            preferred_files = [
                metadata_path.parent / "primary_document.md",
                metadata_path.parent / "primary_document.html",
                metadata_path.parent / "primary_document.txt",
                metadata_path.parent / "full_submission.txt",
            ]
            for preferred in preferred_files:
                if preferred.exists():
                    relative_preferred = str(preferred).replace(str(root / "workspace") + "\\", "").replace(str(root / "workspace") + "/", "")
                    relative_preferred = relative_preferred.replace(str(root) + "\\", "").replace(str(root) + "/", "")
                    relative_metadata = str(metadata_path).replace(str(root / "workspace") + "\\", "").replace(str(root / "workspace") + "/", "")
                    relative_metadata = relative_metadata.replace(str(root) + "\\", "").replace(str(root) + "/", "")
                    register(
                        _candidate_from_local_path(
                            run_input=run_input,
                            local_abs=preferred,
                            relative_path=relative_preferred.replace("\\", "/"),
                            discovered_by="edgar_download",
                            source_artifact_path=relative_metadata.replace("\\", "/"),
                            reasons=["Downloaded Edgar filing artifact selected for unstructured ingestion."],
                            metadata=metadata,
                            source_type="edgar_filing",
                        )
                    )
                    break

    raw_candidates = sorted(candidates_by_key.values(), key=lambda item: item.candidate_id)
    vetted_candidates = await vet_stage_candidates(
        mission_id=run_input.mission_id,
        task_slug=run_input.task_slug,
        user_objective=run_input.user_objective,
        targets=run_input.targets,
        candidates=raw_candidates,
    )

    manifest = StageCandidateManifest(
        mission_id=run_input.mission_id,
        task_slug=run_input.task_slug,
        candidate_dir=relative_to_root(candidate_dir, root=root),
        raw_candidate_count=len(raw_candidates),
        candidate_count=len(vetted_candidates),
        candidates=vetted_candidates,
        skipped_items=skipped_items,
    )
    manifest_path = candidate_dir / "stage_candidate_manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest, relative_to_root(manifest_path, root=root)


def gather_mission_candidates(
    *,
    mission_id: str,
    stage_manifest_paths: list[str],
    root: Path = ROOT_FILESYSTEM,
) -> tuple[MissionCandidateManifest, str]:
    candidates_by_key: dict[str, CandidateDocument] = {}
    dedupe_notes: list[str] = []
    total_stage_candidates = 0

    for rel_path in stage_manifest_paths:
        manifest_abs = root / rel_path
        payload = _load_json_if_exists(manifest_abs)
        if payload is None:
            dedupe_notes.append(f"Skipped unreadable stage manifest: {rel_path}")
            continue
        stage_manifest = StageCandidateManifest.model_validate(payload)
        total_stage_candidates += stage_manifest.candidate_count
        for candidate in stage_manifest.candidates:
            existing = candidates_by_key.get(candidate.dedupe_key)
            if existing is None:
                candidates_by_key[candidate.dedupe_key] = candidate
                continue
            dedupe_notes.append(
                f"Deduped candidate `{candidate.title or candidate.uri or candidate.local_path}` "
                f"across stages `{existing.provenance.task_slug}` and `{candidate.provenance.task_slug}`."
            )
            existing.reasons = sorted(set(existing.reasons + candidate.reasons))
            existing.metadata.update(candidate.metadata)
            if not existing.local_path and candidate.local_path:
                existing.local_path = candidate.local_path
                existing.relative_path = candidate.relative_path
            if not existing.uri and candidate.uri:
                existing.uri = candidate.uri

    mission_dir = mission_candidate_dir(mission_id, root=root)
    manifest = MissionCandidateManifest(
        mission_id=mission_id,
        stage_manifest_paths=stage_manifest_paths,
        total_stage_candidates=total_stage_candidates,
        final_candidate_count=len(candidates_by_key),
        candidates=sorted(candidates_by_key.values(), key=lambda item: item.candidate_id),
        dedupe_notes=dedupe_notes,
    )
    manifest_path = mission_dir / "mission_candidate_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest, relative_to_root(manifest_path, root=root)
