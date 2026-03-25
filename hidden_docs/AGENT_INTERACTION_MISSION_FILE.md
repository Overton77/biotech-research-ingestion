### Surface 2 — Execution (Shell → `uv run`)

All commands run from: `biotech-research-ingestion/`

#### Mission runs (primary test surface)

```bash
# Named mission (hardcoded in models/mission.py)
uv run python -m src.research.langchain_agent.run_mission --mission qualia
uv run python -m src.research.langchain_agent.run_mission --mission elysium

# JSON mission file — the preferred way to test new research missions
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/thorne_research.json \
  --output-dir   src/research/langchain_agent/test_runs/run_outputs/thorne_research

# Run only one stage from a mission file (useful for debugging a single stage)
uv run python -m src.research.langchain_agent.run_mission \
  --mission-file src/research/langchain_agent/test_runs/missions/thorne_research.json \
  --stage thorne-company-fundamentals
```

#### KG ingestion

```bash
# Dry run first, then live
uv run python -m src.research.langchain_agent.kg.ingest_report \
  --report reports/elysium-products-and-specs.md \
  --targets "Elysium Health" \
  --dry_run

uv run python -m src.research.langchain_agent.kg.ingest_report \
  --report reports/elysium-products-and-specs.md \
  --targets "Elysium Health"

# Batch ingest all reports
uv run python -m src.research.langchain_agent.scripts.ingest_all_reports
```

#### Inspection scripts (planned — not yet created)

```bash
uv run python -m src.research.langchain_agent.scripts.check_neo4j
uv run python -m src.research.langchain_agent.scripts.run_single_stage \
  --stage qualia-company-fundamentals
```

#### LangSmith evaluation (planned — not yet created)

```bash
uv run python -m src.research.langchain_agent.eval.build_datasets
uv run python -m src.research.langchain_agent.eval.run_eval \
  --dataset biotech-research-reports-v1 \
  --experiment baseline
```

#### Tests (planned — not yet created)

```bash
uv run pytest src/research/langchain_agent/tests/ -v
```

### Surface 3 — Output Retrieval (five channels)

| Channel           | How                                                                     | What I retrieve                                            |
| ----------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Agent files**   | `Read` tool on `agent_outputs/reports/*.md` and `agent_outputs/runs/*/` | Final reports, intermediate reasoning files                |
| **Neo4j (MCP)**   | `user-neo4j-dev-dev-read_neo4j_cypher`                                  | Node/relationship counts, entity data, vector index status |
| **MongoDB (MCP)** | `user-MongoDB-find` / `user-MongoDB-aggregate`                          | Research run documents, Beanie episode records             |
| **LangSmith**     | Browser → `smith.langchain.com` or `eval/run_eval.py` stdout            | Trace trees, evaluation scores, experiment comparisons     |
| **Terminal**      | Shell stdout + `.cursor/terminals/*.txt`                                | Runtime logs, errors, timing                               |

### Surface 3b — JSON Mission Test Apparatus (active)

This is the primary hands-on testing surface. Write a mission file → run it → inspect outputs.

#### Directory layout

```
src/research/langchain_agent/
├── test_runs/
│   ├── missions/                    ← JSON mission definitions go here
│   │   ├── elysium_mini.json        ← single-stage smoke test (search_web only)
│   │   ├── qualia_mini.json         ← single-stage smoke test (search_web only)
│   │   └── thorne_research.json     ← full 2-stage mission (first live run: 2026-03-21)
│   │
│   └── run_outputs/                 ← one subdirectory per mission run
│       └── thorne_research/
│           ├── mission_summary.json              ← written at end (status, all stage records)
│           ├── stage_01_thorne-company-fundamentals.json      ← per-stage JSON record
│           ├── stage_01_thorne-company-fundamentals_report.md ← final report text
│           ├── stage_02_thorne-products-research.json
│           └── stage_02_thorne-products-research_report.md
│
└── agent_outputs/                   ← agent's own sandbox (written during run)
    ├── reports/                     ← final .md reports (relative path: reports/<slug>.md)
    └── runs/                        ← intermediate checkpoint files per stage
        ├── thorne-company-fundamentals/
        │   ├── 01_discovery_notes.md
        │   └── 02_leadership_notes.md
        └── thorne-products-research/
            ├── 01_url_map.md
            ├── 02_flagship_products.md
            ├── 03_sports_products.md
            ├── 04_foundational_products.md
            └── 05_specialty_products.md
```
