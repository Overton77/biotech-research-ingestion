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


class ExtractedOrganization(BaseModel):
    name: str
    aliases: list[str] = []
    orgType: str = "COMPANY"
    businessModel: str = "B2C"
    description: str = ""
    legalName: str = ""
    websiteUrl: str = ""
    primaryIndustryTags: list[str] = []
    regionsServed: list[str] = []
    headquartersCity: str = ""
    headquartersCountry: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity's state, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'aliases', 'description', 'businessModel', 'primaryIndustryTags'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedPerson(BaseModel):
    canonicalName: str
    givenName: str = ""
    familyName: str = ""
    honorific: str = ""
    bio: str = ""
    primaryDomain: str = ""
    specialties: list[str] = []
    expertiseTags: list[str] = []
    degrees: list[str] = []
    linkedinUrl: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity's state, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['canonicalName', 'bio', 'primaryDomain', 'specialties', 'expertiseTags'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedProduct(BaseModel):
    name: str
    synonyms: list[str] = []
    productDomain: str = "SUPPLEMENT"
    productType: str = ""
    brandName: str = ""
    priceAmount: float | None = None
    currency: str = "USD"
    intendedUse: str = ""
    description: str = ""
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this entity's state, if available in the report.",
    )
    searchFields: list[str] = Field(
        default=['name', 'synonyms', 'brandName', 'intendedUse', 'description'],
        description="Fields used for searchText generation — do not alter.",
    )


class ExtractedOrgPersonRelationship(BaseModel):
    org_name: str
    person_name: str
    relationship_type: str
    roleTitle: str = ""
    department: str = ""
    seniority: str = ""
    isCurrent: bool = True
    temporal: Optional[TemporalQualifier] = Field(
        default=None,
        description="Temporal evidence for this relationship, if available (e.g. 'joined 2020', 'left Q1 2024').",
    )
    searchFields: list[str] = Field(
        default=['roleTitle', 'department', 'seniority'],
        description="Fields used for searchText generation — do not alter.",
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
        description="Temporal evidence for this ingredient relationship, if available.",
    )
    searchFields: list[str] = Field(
        default=['compoundName', 'formType', 'bioavailabilityNotes'],
        description="Fields used for searchText generation — do not alter.",
    )


class KGExtractionResult(BaseModel):
    """Top-level extraction output returned by the extraction agent."""

    source_report: str = ""
    organizations: list[ExtractedOrganization] = []
    persons: list[ExtractedPerson] = []
    products: list[ExtractedProduct] = []
    compound_ingredients: list[ExtractedCompoundIngredient] = []
    org_person_relationships: list[ExtractedOrgPersonRelationship] = []
