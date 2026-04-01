# AUTO-GENERATED — do not edit directly.
# Source of truth: schema/schema_registry.json
# Regenerate:  python -m src.research.langchain_agent.kg.codegen_extraction_models
"""
Pydantic models for KG entity extraction output.

All entity types the LLM may extract from a research report.
searchFields on each model drives deterministic searchText generation;
the LLM should not alter this list.

Temporal qualifiers: each entity and relationship carries optional
temporal_qualifier and temporal_context fields.  The LLM fills these
when the report provides explicit temporal evidence (e.g. "as of 2024",
"since March 2023", "formerly known as").

Generated from: schema/schema_registry.json
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Temporal context (passed INTO extraction, not extracted BY LLM)
# ---------------------------------------------------------------------------


class TemporalScope(BaseModel):
    """
    Temporal scope carried by the research configuration.

    Tells the extraction system what time frame the research covers.
    The LLM extraction prompt includes this so temporal reasoning is grounded.
    """

    mode: Literal["current", "as_of_date", "date_range", "unknown"] = "current"
    as_of_date: Optional[str] = Field(
        default=None,
        description="ISO date string (YYYY-MM-DD) when mode='as_of_date'.",
    )
    range_start: Optional[str] = Field(
        default=None,
        description="ISO date string for range start when mode='date_range'.",
    )
    range_end: Optional[str] = Field(
        default=None,
        description="ISO date string for range end when mode='date_range'.",
    )
    description: str = Field(
        default="Current state as of research date.",
        description="Human-readable description of the temporal scope.",
    )


class IngestionTemporalContext(BaseModel):
    """
    System-level temporal context passed through the ingestion pipeline.
    Not produced by the LLM — set by the orchestrator or coordinator.
    """

    research_date: Optional[datetime] = Field(
        default=None,
        description="When the research was conducted. Used as validFrom default.",
    )
    ingestion_time: Optional[datetime] = Field(
        default=None,
        description="When the ingestion run started. Used as recordedFrom.",
    )
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    source_report: str = ""


# ---------------------------------------------------------------------------
# Temporal qualifier (extracted BY the LLM when evidence exists)
# ---------------------------------------------------------------------------


class TemporalQualifier(BaseModel):
    """Optional temporal evidence extracted from the report text."""

    valid_from: Optional[str] = Field(
        default=None,
        description="ISO date or descriptive string (e.g. '2023-01', 'founded 2014'). When the fact became true.",
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="ISO date or descriptive string. When the fact ceased being true (null = still active).",
    )
    temporal_note: str = Field(
        default="",
        description="Free-text temporal context from the report, e.g. 'as of Q2 2025', 'since founding'.",
    )


# ---------------------------------------------------------------------------
# Extracted entities
# ---------------------------------------------------------------------------


class ExtractedBiomarker(BaseModel):
    name: str
    description: str = ""
    biomarkerType: str = ""
    specimenMatrix: str = ""
    moleculeClass: str = ""
    clinicalSignificance: str = ""
    commonUnits: list[str] = []
    agingHallmark: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedCompound(BaseModel):
    name: str
    description: str = ""
    commonName: str = ""
    casNumber: str = ""
    molecularFormula: str = ""
    molecularWeight: float | None = None
    compoundClass: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description', 'commonName'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedCondition(BaseModel):
    name: str
    description: str = ""
    conditionClass: str = ""
    mondoId: str = ""
    icd11Code: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedLabTest(BaseModel):
    name: str
    description: str = ""
    testType: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedOrganization(BaseModel):
    name: str
    description: str = ""
    canonicalTicker: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedPanelDefinition(BaseModel):
    name: str
    description: str = ""
    panelType: str = ""
    versionLabel: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedPerson(BaseModel):
    name: str
    description: str = ""
    title: str = ""
    bio: str = ""
    linkedInUrl: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description', 'bio', 'title'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedProduct(BaseModel):
    name: str
    description: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedStudy(BaseModel):
    name: str
    description: str = ""
    registryNamespace: str = ""
    registryId: str = ""
    canonicalUrl: str = ""
    pmid: str = ""
    doi: str = ""
    overallStatus: str = ""
    startDate: str = ""
    enrollmentCount: int | None = None
    studyType: str = ""
    countries: list[str] = []
    hasResults: bool | None = None
    studyPhase: str = ""
    sampleSizeText: str = ""
    evidenceLevel: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedLabTestBiomarkerRelationship(BaseModel):
    lab_test_name: str
    biomarker_name: str
    role: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedOrgPersonRelationship(BaseModel):
    org_name: str
    person_name: str
    relationship_type: str
    roleTitle: str = ""
    department: str = ""
    seniority: str = ""
    isCurrent: bool | None = True
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedOrgProductRelationship(BaseModel):
    org_name: str
    product_name: str
    relationship_type: str
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedCompoundIngredient(BaseModel):
    product_name: str
    compoundName: str
    formType: str = ""
    dose: float | None = None
    doseUnit: str = ""
    role: str = "active"
    bioavailabilityNotes: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )
    searchFields: list[str] = Field(
        default=['compoundName', 'formType'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedProductLabTestRelationship(BaseModel):
    product_name: str
    lab_test_name: str
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedProductPanelRelationship(BaseModel):
    product_name: str
    panel_name: str
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedStudyConditionRelationship(BaseModel):
    study_name: str
    condition_name: str
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedStudyOrgRelationship(BaseModel):
    study_name: str
    org_name: str
    relationship_type: str
    role: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class ExtractedStudyPersonRelationship(BaseModel):
    study_name: str
    person_name: str
    role: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available.",
    )


class KGExtractionResult(BaseModel):
    """Top-level extraction output returned by the extraction agent."""

    source_report: str = ""
    biomarkers: list[ExtractedBiomarker] = []
    compounds: list[ExtractedCompound] = []
    conditions: list[ExtractedCondition] = []
    labTests: list[ExtractedLabTest] = []
    organizations: list[ExtractedOrganization] = []
    panelDefinitions: list[ExtractedPanelDefinition] = []
    persons: list[ExtractedPerson] = []
    products: list[ExtractedProduct] = []
    studies: list[ExtractedStudy] = []
    compound_ingredients: list[ExtractedCompoundIngredient] = []
    lab_test_biomarker_relationships: list[ExtractedLabTestBiomarkerRelationship] = []
    org_person_relationships: list[ExtractedOrgPersonRelationship] = []
    org_product_relationships: list[ExtractedOrgProductRelationship] = []
    product_lab_test_relationships: list[ExtractedProductLabTestRelationship] = []
    product_panel_relationships: list[ExtractedProductPanelRelationship] = []
    study_condition_relationships: list[ExtractedStudyConditionRelationship] = []
    study_org_relationships: list[ExtractedStudyOrgRelationship] = []
    study_person_relationships: list[ExtractedStudyPersonRelationship] = []
