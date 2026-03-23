# ResearchMissionDraft Output Schema

You MUST return a JSON object conforming exactly to this schema. No extra fields are allowed.

## Top-level fields

```json
{
  "title": "string — concise mission title",
  "goal": "string — the overarching research goal",
  "global_context": {
    "domain": "string — e.g. 'biotech', 'pharma'",
    "key_terms": ["string — domain-specific terms relevant to all tasks"]
  },
  "global_constraints": ["string — constraints that apply to all tasks"],
  "success_criteria": ["string — criteria for mission success"],
  "task_defs": [ ... ]
}
```

## TaskDef schema (each element of task_defs)

```json
{
  "task_id": "string — MUST match a task id from the original plan",
  "name": "string — human-readable task name",
  "stage_label": "string | null — stage name from the plan",
  "description": "string — what this task accomplishes",
  "depends_on": ["string — task_ids this task depends on (from the plan)"],
  "input_bindings": {
    "local_input_name": {
      "source_task_id": "string — which prior task provides this input",
      "source_key": "string — which output key from that task (typically 'response' or 'files')",
      "required": true,
      "transform": "null | 'summarize' — if 'summarize', an LLM condenses the input; otherwise plain truncation",
      "max_tokens": "int | null — if set, large inputs are truncated/summarized to this token budget"
    }
  },
  "output_schema": {
    "response": "string — primary text output",
    "files": "list — artifact files produced"
  },
  "acceptance_criteria": ["string — criteria the output will be judged against"],
  "main_agent": {
    "model_name": "openai:gpt-5",
    "model_tier": "fast | standard | powerful — defaults to 'standard'. Use 'fast' for simple tasks, 'powerful' for complex synthesis",
    "system_prompt": "string — 2-4 paragraph detailed system prompt. MUST include brief instructions to update AGENTS.md and write important memories and output paths to /memories/.",
    "tool_profile_name": "default_research | search_only | write_only",
    "filesystem_profile": "task_local",
    "memory_profile": "in_memory",
    "allow_general_purpose_subagent": true,
    "max_iterations": null,
    "skills": ["string — skill directory paths, e.g. '/skills/source-citation/'"],
    "notes": {}
  },
  "compiled_subagents": [
    {
      "name": "string — unique subagent name (e.g. 'source-finder', 'evidence-extractor')",
      "description": "string — what the main agent uses to decide when to delegate",
      "system_prompt": "string — detailed instructions for this subagent",
      "model_name": null,
      "tool_profile_name": "default_research | search_only | write_only",
      "filesystem_profile": "subagent_local",
      "use_todo_middleware": false,
      "memory_profile": "in_memory",
      "workspace_suffix": "string — unique folder name (e.g. 'source_finder', 'evidence_extractor')",
      "max_invocations": 1,
      "expected_output_format": "string | null — optional; description of expected content/format (e.g. 'Markdown table of products: name, category, link')",
      "expected_output_path": "string | null — optional; path relative to subagent workspace (e.g. 'outputs/company_products.md')",
      "skills": ["string — skill directory paths for this subagent, e.g. '/skills/source-citation/'"]
    }
  ],
  "execution": {
    "timeout_seconds": 300,
    "max_retries": 1,
    "persist_run_after_completion": true,
    "fallback_models": ["string — optional fallback model names, e.g. 'openai:gpt-5', 'anthropic:claude-sonnet-4-20250514'"],
    "require_human_review": false
  }
}
```

## Critical Rules

1. **task_id** values MUST exactly match task IDs from the original research plan.
2. **depends_on** MUST only reference task_ids that exist in the plan — do not invent dependencies.
3. Each task MUST have 1-3 compiled_subagents with distinct roles and specific system_prompts.
4. Main agent system_prompts MUST include explicit instructions on when to delegate to each subagent by name.
5. **Each main_agent system_prompt MUST include guidance on updating AGENTS.md and writing to /memories/** (important memories, intermediate and final output paths).
6. input_bindings for tasks with dependencies MUST reference valid (source_task_id, source_key) pairs.
7. workspace_suffix values MUST be unique within a single task's compiled_subagents list.
8. Assign `skills: ["/skills/source-citation/"]` to every main_agent and to every subagent whose tool_profile_name is "default_research" or "search_only".
9. For input_bindings on dependent tasks, set `max_tokens` (e.g. 2000) when prior tasks may produce large outputs. Use `transform: "summarize"` for concise context passing.
