# Observability & Evaluation Plan

> **Scope:** `src/research/langchain_agent/`  
> **Created:** 2026-03-21  
> **Status:** Implementation in progress

---

## 1. Architecture Assessment

### What exists today

The research pipeline is well-structured with clear separation of concerns:

| Layer | Files | Responsibility |
|---|---|---|
| **CLI entry** | `run_mission.py` | Arg parsing, mission loading, output writing |
| **Orchestration** | `workflow/run_mission.py` | DAG topological sort, stage sequencing, MongoDB persistence |
| **Execution** | `workflow/run_slice.py` | Single-stage lifecycle: memory recall → agent run → memory ingestion → S3 upload |
| **Iteration** | `workflow/run_iterative_stage.py`, `iteration_control.py` | Multi-pass loop with next-steps extraction and stop conditions |
| **Agent construction** | `agent/factory.py` | `build_research_agent()`, `build_memory_report_agent()`, `build_next_steps_agent()` |
| **Persistence** | `storage/models.py`, `s3_store.py`, `langgraph_persistence.py` | Beanie docs, S3 artifacts, Postgres checkpointer/store |

### What works well

1. **LangSmith tracing is already enabled** — `.env` has `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT="biotech_research"`. All `create_agent()` calls auto-emit traces.
2. **Clean execution lifecycle** — `run_single_mission_slice()` is the universal unit of work for both stage-based and iterative missions.
3. **Structured outputs** — `StageRunRecord`, `MissionRunDocument`, and `NextStepsArtifact` provide rich structured data.
4. **Dependency wiring** — report injection between stages is clean and deterministic.

### Gaps preventing full observability and evaluation

| # | Gap | Impact | Severity |
|---|---|---|---|
| G1 | **No trace hierarchy** — each `create_agent()` call emits an independent trace. There is no parent span linking stages to a mission, or iterations to a stage. | Cannot view a mission as a single trace tree. Stages appear as disconnected runs in LangSmith. | High |
| G2 | **No metadata/tags on traces** — no `mission_id`, `task_slug`, `stage_type`, `iteration` attached to LangSmith runs. | Cannot filter, search, or group traces by mission or stage. | High |
| G3 | **No `@traceable` on orchestration functions** — `run_mission()`, `run_single_mission_slice()`, `_extract_next_steps()`, memory recall, and memory ingestion are invisible in traces. | ~60% of the execution lifecycle is not captured. Only the LLM calls inside `create_agent()` are visible. | High |
| G4 | **No evaluation framework** — no datasets, evaluators, or experiment infrastructure. | Cannot measure quality, detect regressions, or compare runs. | High |
| G5 | **No run-level metadata persistence** — LangSmith trace IDs are not stored in `StageRunRecord` or `MissionRunDocument`. | Cannot link MongoDB records to LangSmith traces for cross-referencing. | Medium |
| G6 | **No project segmentation** — all traces go to a single `biotech_research` project. | Dev/test/eval traces are mixed with production runs. | Medium |
| G7 | **Console `print()` for observability** — runtime progress is only visible via stdout. | No structured logging that could feed into monitoring. | Low |

---

## 2. Implementation Plan

### Phase A — Tracing Infrastructure (highest leverage)

**Goal:** Make every research mission fully traceable as a single hierarchical trace in LangSmith.

#### A1. Add `@traceable` to orchestration functions

Wrap the key functions with `@traceable` to create a proper trace hierarchy:

```
Mission trace (run_mission)
  ├── Stage trace (run_single_mission_slice) — with metadata
  │   ├── Memory recall (load_memories_for_prompt)
  │   ├── Research agent (create_agent.ainvoke) — auto-traced
  │   ├── Memory report agent (create_agent.ainvoke) — auto-traced
  │   └── Memory ingestion (memory_manager.ainvoke)
  │
  ├── Iterative stage trace (run_iterative_stage)
  │   ├── Iteration 1 trace (run_single_mission_slice)
  │   │   └── ... (same as above)
  │   ├── Next-steps extraction (_extract_next_steps)
  │   ├── Iteration 2 trace (run_single_mission_slice)
  │   │   └── ...
  │   └── Next-steps extraction
  │
  └── Stage trace (run_single_mission_slice)
```

**Files to modify:**
- `workflow/run_mission.py` — wrap `run_mission()`
- `workflow/run_slice.py` — wrap `run_single_mission_slice()`
- `workflow/run_iterative_stage.py` — wrap `run_iterative_stage()` and `_extract_next_steps()`
- `run_mission.py` — wrap `main()` as the root trace

#### A2. Attach metadata and tags to every trace

Every traced function should carry:
- `mission_id` — links all traces for one mission
- `task_slug` — identifies the stage
- `stage_type` — discovery, entity_validation, etc.
- `iteration` — cycle number for iterative stages (when applicable)

Tags for filtering:
- `mission-type:stage_based` or `mission-type:iterative`
- `stage:<task_slug>`
- `env:dev` / `env:eval` / `env:production`

#### A3. Store LangSmith trace IDs in persistence records

After each traced function completes, capture `get_current_run_tree().id` and store it in:
- `StageRunRecord.langsmith_run_id`
- `MissionRunDocument.langsmith_run_id`

This creates a bidirectional link: MongoDB → LangSmith and LangSmith → MongoDB (via metadata).

#### A4. Project segmentation via `tracing_context`

Use `langsmith.tracing_context` to route traces to the right project:

| Context | Project |
|---|---|
| CLI mission runs | `biotech-research-missions` |
| Evaluation runs | `biotech-research-evals` |
| Development/debugging | `biotech-research-dev` |

The default `LANGSMITH_PROJECT` stays as `biotech_research` for backward compatibility. Specific contexts override it.

### Phase B — Evaluation Framework

**Goal:** Build a practical evaluation pipeline that supports local iteration and production-grade comparison.

#### B1. Dataset creation from existing outputs

Build `eval/build_datasets.py` that:
1. Reads completed reports from `agent_outputs/reports/` and `test_runs/run_outputs/`
2. Reads the corresponding mission JSON to reconstruct inputs
3. Creates LangSmith datasets with `inputs` (mission config) and `outputs` (report text, structured data)

Dataset types:
- **`biotech-research-reports-v1`** — full mission inputs → final report text
- **`biotech-stage-outputs-v1`** — per-stage inputs → stage report + artifacts

#### B2. Evaluators

Build `eval/evaluators.py` with these evaluator functions:

| Evaluator | Type | What it measures |
|---|---|---|
| `report_completeness` | LLM-as-judge | Does the report cover all required sections? |
| `report_accuracy` | LLM-as-judge | Are claims supported by cited sources? |
| `report_structure` | Code | Are all `report_required_sections` present? |
| `source_quality` | Code | Number of unique sources, domain diversity |
| `tool_efficiency` | Code | Steps used vs. budget, tool call success rate |
| `iteration_convergence` | Code | For iterative stages: confidence trend, diminishing returns detection |

#### B3. Run function and experiment runner

Build `eval/run_eval.py` that:
1. Loads a dataset from LangSmith
2. Defines a `run_function` that executes a mission stage and returns outputs
3. Calls `langsmith.aevaluate()` with the evaluators
4. Tags the experiment with model version, prompt version, and timestamp

```bash
uv run python -m src.research.langchain_agent.eval.run_eval \
  --dataset biotech-research-reports-v1 \
  --experiment baseline-gpt5mini \
  --model gpt-5-mini
```

#### B4. Comparison workflow

Build `eval/compare.py` for pairwise evaluation:
1. Run the same dataset with two different configurations (model, prompt, tools)
2. Use LangSmith pairwise evaluation to rank outputs
3. Generate a comparison report

### Phase C — Integration and Hardening

#### C1. Structured logging

Replace `print()` calls with structured `logger.info()` that includes mission/stage context. This feeds into both terminal output and future monitoring.

#### C2. Neo4j integration readiness

Ensure the tracing and evaluation design accommodates:
- KG ingestion as a traced post-stage step
- Schema selection as a traced decision point
- Neo4j write verification as an evaluator dimension

#### C3. GraphQL API readiness

The evaluation framework should be extensible to evaluate:
- GraphQL query accuracy (when agents use KG tools)
- Data completeness (comparing agent-extracted data vs. KG state)

---

## 3. File Layout

```
src/research/langchain_agent/
├── observability/                    ← NEW: tracing utilities
│   ├── __init__.py
│   └── tracing.py                   # @traceable wrappers, metadata helpers, project routing
│
├── eval/                            ← NEW: evaluation framework
│   ├── __init__.py
│   ├── build_datasets.py            # Create LangSmith datasets from existing outputs
│   ├── evaluators.py                # All evaluator functions (LLM-as-judge + code)
│   ├── run_eval.py                  # CLI: run evaluation against a dataset
│   ├── compare.py                   # Pairwise comparison between experiments
│   └── rubrics.py                   # Scoring criteria constants
│
├── workflow/
│   ├── run_mission.py               # MODIFIED: add @traceable, metadata, trace ID capture
│   ├── run_slice.py                 # MODIFIED: add @traceable, metadata, trace ID capture
│   └── run_iterative_stage.py       # MODIFIED: add @traceable, metadata
│
├── storage/
│   └── models.py                    # MODIFIED: add langsmith_run_id fields
│
└── run_mission.py                   # MODIFIED: root trace, project routing
```

---

## 4. Implementation Priority

| Priority | Item | Effort | Impact |
|---|---|---|---|
| **P0** | A1: `@traceable` on orchestration functions | 2h | Unlocks full trace visibility |
| **P0** | A2: Metadata and tags on traces | 1h | Enables filtering and grouping |
| **P1** | A3: Trace ID in persistence records | 30m | Bidirectional linking |
| **P1** | B1: Dataset creation from existing outputs | 2h | Foundation for all evaluation |
| **P1** | B2: Core evaluators (completeness, structure, sources) | 3h | Quality measurement |
| **P2** | B3: Experiment runner CLI | 2h | Repeatable evaluation |
| **P2** | A4: Project segmentation | 30m | Clean trace organization |
| **P3** | B4: Pairwise comparison | 2h | Regression detection |
| **P3** | C1: Structured logging | 1h | Better runtime visibility |

---

## 5. Design Decisions

### Why `@traceable` over `RunTree` API?

`@traceable` is the right choice because:
1. The codebase is fully `async` — `@traceable` handles async natively
2. Python 3.12+ has full `contextvars` support — parent/child nesting is automatic
3. `create_agent()` already emits traces — `@traceable` wrappers nest correctly as parents
4. Minimal code changes — decorators are additive, not invasive

### Why not a separate tracing middleware?

The orchestration layer (`run_mission` → `run_slice`) is not middleware-compatible — it's plain async functions. `@traceable` is the natural fit. The agent-level tracing (inside `create_agent`) is already handled by LangChain's built-in callbacks.

### Why build evaluators locally first?

LangSmith uploaded evaluators run in a sandboxed environment with limited package access. For our evaluators (which need access to mission models, file I/O, and potentially Neo4j), local `evaluate()` with passed evaluator functions is more practical. We can upload simpler code evaluators later for auto-run on datasets.

### How does this fit with Temporal (future)?

When Temporal wraps `run_single_mission_slice` as an Activity:
- The `@traceable` decorator stays on the function — it works inside Activities
- Temporal's own observability (workflow history, activity retries) complements LangSmith traces
- The `langsmith_run_id` stored in `StageRunRecord` can be correlated with Temporal activity IDs

### How does this fit with Neo4j GraphQL (future)?

- KG ingestion steps will get their own `@traceable` spans
- GraphQL tool calls (when agents query the KG) will be auto-traced as tool calls
- Evaluation can include a "KG accuracy" dimension comparing extracted entities vs. ground truth

---

## 6. Environment Setup

Already configured in `.env`:
```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT="biotech_research"
```

Additional package needed:
```bash
uv add openevals  # prebuilt evaluator prompts
```

No other dependencies required — `langsmith` is already in `pyproject.toml`.
