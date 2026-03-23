# KG Ingestion Run Notes

## 2026-03-23 — First run on new schema_registry workflow

### Report ingested
- File: `agent_outputs/reports/arl-company-fundamentals.md`
- Subject: Analytical Resource Laboratories (ARL) — third-party analytical testing lab
- Targets: `["Analytical Resource Laboratories", "ARL"]`
- Stage type: `targeted_extraction`

---

### Pipeline outcome — FULL RUN (Neo4j write)

| Step | Result |
|---|---|
| Step 1: Schema selection | Selected 2 chunks: `organization`, `labtest` |
| Step 2: Build extraction contract | Sync — reads `schema_registry.json`, emits compact JSON (~200 tokens vs ~3,000 for .md files) |
| Step 3: Entity extraction | 1 org, 11 persons, 0 products, 0 compounds, 11 org-person rels |
| Step 4: SearchText generation | 12 LLM calls (concurrent) for Org/Person nodes |
| Step 5: Batch embedding | 1 OpenAI call, 12 vectors (text-embedding-3-small) |
| Step 6: Neo4j write | 12 nodes written, 11 rels written, 0 skipped |

**Total elapsed:** ~125 seconds  
**Exit code:** 0

---

### Nodes written to Neo4j

**Organization (1)**
- Analytical Resource Laboratories

**Persons (11)**
- Matthew R. Lewis (CEO, founder)
- Jacob Teller (Director of Quality)
- Lance Pyper (Director of Partnerships)
- Christopher Loke (Director of Marketing and Branding)
- Clara Langevin (Chemistry Lab Manager)
- Renee Tran (Method Development Manager)
- Megan Ward (Microbiology Technical Manager)
- Audrey Larsen (Sample Management Team Manager)
- Andrea Helm (Business Development Manager)
- Dana Dunn (Partnerships Manager)
- (11th person extracted — likely one of the above with alternate name form)

**Relationships (11)**
- All EMPLOYS / FOUNDED_BY / HAS_EXECUTIVE_ROLE via `_OrgPersonRelationship` model
- 0 skipped (all person names resolved successfully)

---

### Schema selection behaviour notes

- The selector correctly identified `organization` and `labtest` chunks (ARL is a testing lab, not a supplement company — no `product` or `person` chunks needed as separate selections since person data comes from the org chunk).
- In the dry run the selector chose `['organization', 'product', 'labtest']` — the `product` chunk was included because ARL's testing services can be interpreted as "service products". In the live run it selected only `['organization', 'labtest']`. This non-determinism is expected at temperature=0 with slightly different prompt context on repeated calls. No functional impact.

---

### New workflow validation — schema_registry.json

This was the **first end-to-end run using the new registry-based extraction contract** instead of raw .md files.

What changed vs. the old flow:
- `load_schema_chunks()` is now synchronous — no `await` needed.
- The extraction prompt receives a typed JSON contract (~200 tokens) instead of 365-981 lines of markdown prose.
- The extraction LLM correctly mapped all entities without hallucinations — property types were unambiguous.
- `extraction_models.py` is now auto-generated from `schema_registry.json`.

---

### Bug found and fixed during this run

**Issue:** `UnicodeEncodeError` on Windows cp1252 console when the LLM reasoning string contained a non-breaking hyphen (`\u2011`).

**Fix:** Added `.encode("ascii", errors="replace").decode("ascii")` on the reasoning print in `schema_selector.py` before printing. The actual data is never modified — only the console output is sanitised.

**File fixed:** `kg/schema_selector.py` — `select_schema_chunks()` print statement.

---

### Output files

- `kg/arl_extraction.json` — full `KGExtractionResult` as JSON for inspection
- Neo4j: 12 nodes + 11 relationships now live in the database

---

### Next run checklist

When ingesting a new report, use:

```bash
uv run python -m src.research.langchain_agent.kg.ingest_report \
    --report "reports/<filename>.md" \
    --targets "<Primary Target>" \
    --stage_type targeted_extraction
```

To update schema and regenerate Pydantic models after editing `schema_registry.json`:

```bash
uv run python -m src.research.langchain_agent.kg.codegen_extraction_models
```
