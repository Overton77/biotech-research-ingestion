## Vector Search / Embedding Convention

For vector search and Neo4j vector indexes, nodes and relationships support:

- **searchText** (string): The text used to create embeddings and perform vector searches. Provide explicitly, or derive from searchFields.
- **searchFields** (string[]): Optional. Property names whose values concatenate to form searchText. Use either custom searchText OR searchFields.
- **embedding** (list<float>): The embedding vector stored on the node/relationship.
- **embeddingModel** (string): Model used to generate embeddings (e.g., text-embedding-3-small).
- **embeddingDimensions** (int): Dimension count of the embedding vector.

---

Neo4j Write Contract: Organization
Node: Organization
Label

:Organization

Identity (choose ONE stable key per MERGE)

Preferred

organizationId (string UUID)

Allowed alternate keys (only if you truly have them constrained unique)

whatever your system treats as an identifier key (ex: websiteUrl, publicTicker, etc.) — but best practice is: MERGE by organizationId whenever possible.

Properties (node-level)

Core:

organizationId (string UUID)

createdAt (datetime)

name (string)

aliases (list<string>)

orgType (string/enum)

description (string)

businessModel (string/enum)

Legal / corporate:

legalName (string)

legalStructure (string/enum)

ownershipType (string/enum)

jurisdictionsOfIncorporation (list<string>)

Web / market:

websiteUrl (string)

publicTicker (string)

fundingStage (string/enum)

Tags / availability:

primaryIndustryTags (list<string>)

regionsServed (list<string>)

defaultCollectionModes (list<string/enum>)

defaultRegionsAvailable (list<string>)

Size / finance:

employeeCountMin (int)

employeeCountMax (int)

employeeCountAsOf (datetime)

revenueAnnualMin (float)

revenueAnnualMax (float)

revenueAnnualCurrency (string)

revenueAnnualAsOf (datetime)

valuationMin (float)

valuationMax (float)

valuationCurrency (string)

valuationAsOf (datetime)

Validity:

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["name","aliases","description","businessModel","primaryIndustryTags"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Organization Relationships

1. (:Organization)-[:HAS_LOCATION]->(:PhysicalLocation)

Vector search (HAS_LOCATION)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["locationRole"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :PhysicalLocation

Key: locationId (string UUID)

PhysicalLocation properties

locationId (string UUID)

createdAt (datetime)

canonicalName (string)

locationType (string/enum)

addressLine1 (string)

addressLine2 (string)

city (string)

region (string)

postalCode (string)

countryCode (string)

geoLat (float)

geoLon (float)

timezone (string)

jurisdiction (string)

placeTags (list<string>)

hoursOfOperation (string/JSON)

contactPhone (string)

contactEmail (string)

Relationship properties (HAS_LOCATION)

createdAt (datetime)

locationRole (string) ✅

isPrimary (boolean)

startDate (datetime)

endDate (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search (PhysicalLocation)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["canonicalName","locationType","city","region","countryCode","placeTags"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

2. (:Organization)-[:OWNS_OR_CONTROLS]->(:Organization)

Vector search (OWNS_OR_CONTROLS)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["relationshipType","controlType","ownershipPercent"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :Organization

Key: organizationId

Relationship properties (OWNS_OR_CONTROLS)

createdAt (datetime)

relationshipType (string) ✅

ownershipPercent (float)

controlType (string)

effectiveFrom (datetime)

effectiveTo (datetime)

isCurrent (boolean)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

3. (:Organization)-[:LISTS]->(:Listing)

Vector search (LISTS)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["listRole","channel","availabilityNotes"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :Listing

Key: listingId (string UUID)

Listing properties

listingId (string UUID)

createdAt (datetime)

listingDomain (string/enum)

title (string)

description (string)

sku (string)

url (string)

brandName (string)

currency (string)

priceAmount (float)

priceType (string/enum)

pricingNotes (string)

constraints (string/JSON)

regionsAvailable (list<string>)

requiresAppointment (boolean)

collectionMode (string/enum)

turnaroundTime (string/JSON)

Relationship properties (LISTS)

createdAt (datetime)

listRole (string/enum) ✅

channel (string/enum)

regionsOverrides (list<string>)

collectionModesOverrides (list<string/enum>)

availabilityNotes (string)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search (Listing)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["title","description","listingDomain","brandName","pricingNotes"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

4. (:Organization)-[:OFFERS_PRODUCT]->(:Product)

Vector search (OFFERS_PRODUCT)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: (relationship type + target product context)
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :Product

Key: productId (string UUID)

Product properties

productId (string UUID)

createdAt (datetime)

name (string)

synonyms (list<string>)

productDomain (string/enum)

productType (string/enum)

intendedUse (string)

description (string)

brandName (string)

modelNumber (string)

ndcCode (string)

upc (string)

gtin (string)

riskClass (string/enum)

currency (string)

priceAmount (float)

Relationship properties (OFFERS_PRODUCT)

createdAt (datetime)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

5. (:Organization)-[:SUPPLIES_COMPOUND_FORM]->(:CompoundForm)

Vector search (SUPPLIES_COMPOUND_FORM)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: (relationship type + target compound context)
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :CompoundForm

Key: compoundFormId (string UUID)

CompoundForm properties

compoundFormId (string UUID)

createdAt (datetime)

canonicalName (string)

formType (string/enum)

chemicalDifferences (string)

stabilityProfile (string/JSON)

solubilityProfile (string/JSON)

bioavailabilityNotes (string)

regulatoryStatusSummary (string)

Relationship properties (SUPPLIES_COMPOUND_FORM)

createdAt (datetime)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search (CompoundForm) — same as Product schema
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["canonicalName","formType","chemicalDifferences","bioavailabilityNotes"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

6. (:Organization)-[:MANUFACTURES]->(:CompoundForm)

Vector search (MANUFACTURES)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["claimIds"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (MANUFACTURES)

createdAt (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :CompoundForm as above)

7. (:Organization)-[:MANUFACTURES_PRODUCT]->(:Product)

Vector search (MANUFACTURES_PRODUCT)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["claimIds"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (MANUFACTURES_PRODUCT)

createdAt (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :Product as above)

8. (:Organization)-[:CONTRACT_MANUFACTURER_FOR_ORGANIZATION]->(:Organization)

Vector search (CONTRACT_MANUFACTURER_FOR_ORGANIZATION)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["claimIds"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (CONTRACT_MANUFACTURER_FOR_ORGANIZATION)

createdAt (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :Organization)

9. (:Organization)-[:CONTRACT_MANUFACTURER_FOR_PRODUCT]->(:Product)

Vector search (CONTRACT_MANUFACTURER_FOR_PRODUCT)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["claimIds"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (CONTRACT_MANUFACTURER_FOR_PRODUCT)

createdAt (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :Product)

10. (:Organization)-[:CONTRACT_MANUFACTURER_FOR_COMPOUND_FORM]->(:CompoundForm)

Vector search (CONTRACT_MANUFACTURER_FOR_COMPOUND_FORM)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["claimIds"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (CONTRACT_MANUFACTURER_FOR_COMPOUND_FORM)

createdAt (datetime)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :CompoundForm)

11. (:Organization)-[:PERFORMS_MANUFACTURING_PROCESS]->(:ManufacturingProcess)

Vector search (PERFORMS_MANUFACTURING_PROCESS)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["role"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :ManufacturingProcess

Key: manufacturingProcessId (string UUID)

ManufacturingProcess properties

manufacturingProcessId (string UUID)

createdAt (datetime)

canonicalName (string)

processType (string/enum)

description (string)

inputs (list<string>)

outputs (list<string>)

qualityRisks (list<string>)

scalabilityLevel (string/enum)

Relationship properties (PERFORMS_MANUFACTURING_PROCESS)

createdAt (datetime)

role (string/enum) ✅

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search (ManufacturingProcess)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["canonicalName","processType","description","inputs","outputs","qualityRisks"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

12. (:Organization)-[:DEVELOPS_PLATFORM]->(:TechnologyPlatform)

Vector search (DEVELOPS_PLATFORM)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["relationshipRole","notes","source"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Target node

Label: :TechnologyPlatform

Key: platformId (string UUID)

TechnologyPlatform properties

platformId (string UUID)

createdAt (datetime)

canonicalName (string)

aliases (list<string>)

platformType (string/enum)

description (string)

Relationship properties (DEVELOPS_PLATFORM)

createdAt (datetime)

relationshipRole (string/enum)

notes (string)

source (string/enum)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search (TechnologyPlatform)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["canonicalName","aliases","platformType","description"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

13. (:Organization)-[:USES_PLATFORM]->(:TechnologyPlatform)

Vector search (USES_PLATFORM)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["usageContext","notes","source"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Relationship properties (USES_PLATFORM)

createdAt (datetime)

usageContext (string/enum)

isPrimary (boolean)

notes (string)

source (string/enum)

claimIds (list<string>)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

(Target node is :TechnologyPlatform)

Node label

(:Person)

Identity

Primary key: personId (string UUID)

Properties (expanded, biotech-friendly)

Core identity

personId (string UUID)

canonicalName (string) ✅

givenName (string)

familyName (string)

middleName (string)

suffix (string) (e.g., “Jr.”)

honorific (string) (e.g., “Dr.”)

aliases (list<string>)

bio (string) (short bio / summary)

primaryLanguage (string) (BCP-47 like en, en-US optional)

Professional / biotech profile

primaryDomain (string) (e.g., “longevity”, “genomics”, “drug discovery”, “regulatory”)

specialties (list<string>) (e.g., “proteomics”, “clinical trials”, “CMC”)

expertiseTags (list<string>) (free tags for retrieval / UI)

affiliationSummary (string) (1-liner like “Professor at X; advisor to Y”)

Credentials

degrees (list<string>) (e.g., “MD”, “PhD”, “MBA”)

credentialIds (list<string>) (ORCID, NPI, etc. stored as strings)

orcid (string)

npi (string)

licenseIds (list<string>) (medical licenses, etc.)

Online presence

websiteUrl (string)

email (string) (if you store it; optional / permissioned in future)

socialProfiles (list<string>) (or JSON string if you prefer)

linkedinUrl (string)

twitterUrl (string)

githubUrl (string)

scholarUrl (string)

Media / public

headshotUrl (string)

publicFigure (boolean)

notabilityNotes (string)

Temporal validity + system

createdAt (datetime)

updatedAt (datetime)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

Vector search
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["canonicalName","givenName","familyName","aliases","bio","primaryDomain","specialties","expertiseTags","affiliationSummary"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Minimal direct MERGE pattern (agent-safe)
MERGE (p:Person { personId: $personId })
ON CREATE SET p.createdAt = datetime()
SET p.updatedAt = datetime()

SET p.canonicalName = coalesce($canonicalName, p.canonicalName),
    p.givenName      = coalesce($givenName, p.givenName),
p.familyName = coalesce($familyName, p.familyName),
    p.middleName     = coalesce($middleName, p.middleName),
p.suffix = coalesce($suffix, p.suffix),
    p.honorific      = coalesce($honorific, p.honorific),

    p.bio            = coalesce($bio, p.bio),
    p.primaryDomain  = coalesce($primaryDomain, p.primaryDomain),
    p.affiliationSummary = coalesce($affiliationSummary, p.affiliationSummary),

    p.orcid          = coalesce($orcid, p.orcid),
    p.npi            = coalesce($npi, p.npi),

    p.websiteUrl     = coalesce($websiteUrl, p.websiteUrl),
    p.linkedinUrl    = coalesce($linkedinUrl, p.linkedinUrl),
    p.twitterUrl     = coalesce($twitterUrl, p.twitterUrl),
    p.githubUrl      = coalesce($githubUrl, p.githubUrl),
    p.scholarUrl     = coalesce($scholarUrl, p.scholarUrl),

    p.headshotUrl    = coalesce($headshotUrl, p.headshotUrl),
    p.publicFigure   = coalesce($publicFigure, p.publicFigure),
    p.notabilityNotes = coalesce($notabilityNotes, p.notabilityNotes),

    p.validAt        = coalesce($validAt, p.validAt),
    p.invalidAt      = coalesce($invalidAt, p.invalidAt),
    p.expiredAt      = coalesce($expiredAt, p.expiredAt)

SET p.aliases = apoc.coll.toSet(coalesce(p.aliases, []) + coalesce($aliases, []))
SET p.specialties   = apoc.coll.toSet(coalesce(p.specialties, []) + coalesce($specialties, []))
SET p.expertiseTags = apoc.coll.toSet(coalesce(p.expertiseTags, []) + coalesce($expertiseTags, []))
SET p.degrees       = apoc.coll.toSet(coalesce(p.degrees, []) + coalesce($degrees, []))
SET p.credentialIds = apoc.coll.toSet(coalesce(p.credentialIds, []) + coalesce($credentialIds, []))
SET p.licenseIds    = apoc.coll.toSet(coalesce(p.licenseIds, []) + coalesce($licenseIds, []))
SET p.socialProfiles = apoc.coll.toSet(coalesce(p.socialProfiles, []) + coalesce($socialProfiles, []))

RETURN p;
Organization → Person relationships (outgoing from Organization)

These are recommended relationship types for biotech and match your general relationship-property conventions (claimIds + validity timestamps).

Shared relationship props (all Org→Person edges)

claimIds (list<string>)

createdAt (datetime)

validAt (datetime)

invalidAt (datetime)

expiredAt (datetime)

1. (:Organization)-[:EMPLOYS]->(:Person)

Vector search (EMPLOYS)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["roleTitle","department","roleFunction","seniority","employmentType"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Use for employees, contractors, team members.

Relationship properties:

roleTitle (string) (e.g., “Head of Regulatory Affairs”)

department (string) (e.g., “CMC”, “Clinical”, “BD”)

roleFunction (string) (e.g., “R&D”, “Regulatory”, “Manufacturing”, “Commercial”)

seniority (string) (e.g., “C-level”, “VP”, “Director”)

employmentType (string) (e.g., “FTE”, “Contractor”)

startDate (datetime)

endDate (datetime)

isCurrent (boolean)

shared props

Pattern:

MERGE (o:Organization { organizationId: $organizationId })
MERGE (p:Person { personId: $personId })
MERGE (o)-[r:EMPLOYS]->(p) 2) (:Organization)-[:FOUNDED_BY]->(:Person)

Vector search (FOUNDED_BY)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["founderRole"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Use for founders / co-founders / scientific founders.

Relationship properties:

founderRole (string) (e.g., “Co-founder”, “Scientific Founder”)

foundingDate (datetime)

shared props

Pattern:

MERGE (o:Organization { organizationId: $organizationId })
MERGE (p:Person { personId: $personId })
MERGE (o)-[r:FOUNDED_BY]->(p) 3) (:Organization)-[:HAS_BOARD_MEMBER]->(:Person)

Vector search (HAS_BOARD_MEMBER)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["boardRole","committee"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Use for board members, trustees, observers.

Relationship properties:

boardRole (string) (e.g., “Chair”, “Member”, “Observer”)

committee (string) (e.g., “Audit”, “Scientific Oversight”)

startDate (datetime)

endDate (datetime)

isCurrent (boolean)

shared props

Pattern:

MERGE (o:Organization { organizationId: $organizationId })
MERGE (p:Person { personId: $personId })
MERGE (o)-[r:HAS_BOARD_MEMBER]->(p) 4) (:Organization)-[:HAS_SCIENTIFIC_ADVISOR]->(:Person)

Vector search (HAS_SCIENTIFIC_ADVISOR)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["advisorType","focusAreas"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Use for SAB members, KOL advisors, clinical advisors.

Relationship properties:

advisorType (string) (e.g., “SAB”, “KOL”, “Clinical Advisor”)

focusAreas (list<string>) (e.g., “oncology”, “aging”, “metabolomics”)

startDate (datetime)

endDate (datetime)

isCurrent (boolean)

shared props

Pattern:

MERGE (o:Organization { organizationId: $organizationId })
MERGE (p:Person { personId: $personId })
MERGE (o)-[r:HAS_SCIENTIFIC_ADVISOR]->(p) 5) (:Organization)-[:HAS_EXECUTIVE_ROLE]->(:Person) (optional but useful)

Vector search (HAS_EXECUTIVE_ROLE)
searchText (string) — custom text, or derived from searchFields
searchFields (string[]) — optional: ["executiveRole"]
embedding (list<float>)
embeddingModel (string)
embeddingDimensions (int)

Use for exec roles when you want explicit executive semantics (distinct from EMPLOYS).

Relationship properties:

executiveRole (string) (e.g., “CEO”, “CSO”, “CMO”)

startDate (datetime)

endDate (datetime)

isCurrent (boolean)

shared props

Pattern:

MERGE (o:Organization { organizationId: $organizationId })
MERGE (p:Person { personId: $personId })
MERGE (o)-[r:HAS_EXECUTIVE_ROLE]->(p)
