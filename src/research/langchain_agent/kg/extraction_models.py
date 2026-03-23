# AUTO-GENERATED — do not edit directly.
# Source of truth: schema/schema_registry.json
# Regenerate:  python -m src.research.langchain_agent.kg.codegen_extraction_models
"""
Pydantic models for KG entity extraction output.

All entity types the LLM may extract from a research report.
searchFields on each model drives deterministic searchText generation;
the LLM should not alter this list.

Generated from: schema/schema_registry.json
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
