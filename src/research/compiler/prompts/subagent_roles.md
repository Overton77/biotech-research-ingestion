# Subagent Role Catalogue

When designing subagents for research tasks, select from these validated role patterns. You may customize system prompts and tool profiles, but these roles represent proven configurations.

## source_finder
- **Purpose**: Discovers and validates primary sources via targeted web search. Finds authoritative papers, databases, regulatory documents, and institutional pages.
- **Recommended tool_profile_name**: `search_only`
- **use_todo_middleware**: `true` (multi-step source discovery often requires iteration)
- **System prompt guidance**: Instruct to find 5-10 high-quality sources, verify source authority (prefer .gov, .edu, peer-reviewed journals), deduplicate, and write a structured source list to `/outputs/sources.md`.

## evidence_extractor
- **Purpose**: Extracts structured evidence claims from gathered documents. Reads source content, identifies key findings, data points, mechanisms, and writes structured evidence files.
- **Recommended tool_profile_name**: `default_research`
- **use_todo_middleware**: `true` (iterating through multiple sources is multi-step)
- **System prompt guidance**: Instruct to read each source, extract claims with citations, identify confidence levels, flag contradictions, and output a structured evidence file to `/outputs/evidence.json` or `/outputs/evidence.md`.

## verifier
- **Purpose**: Cross-checks key claims against independent sources. Validates facts, identifies conflicting evidence, and assigns confidence scores.
- **Recommended tool_profile_name**: `search_only`
- **use_todo_middleware**: `false` (verification is typically a focused single-pass operation)
- **System prompt guidance**: Instruct to take the evidence set, search for corroborating/contradicting sources for each major claim, and output a verification report to `/outputs/verification.md`.

## synthesizer
- **Purpose**: Integrates evidence from multiple sources into coherent structured summaries. Resolves contradictions, identifies consensus, and produces analytical summaries.
- **Recommended tool_profile_name**: `write_only`
- **use_todo_middleware**: `false` (synthesis is a focused writing operation)
- **System prompt guidance**: Instruct to read all evidence files from the workspace, synthesize into a cohesive narrative with sections, write to `/outputs/synthesis.md`.

## writer
- **Purpose**: Produces the final structured report or document. Formats, edits, and assembles the final deliverable from synthesized content.
- **Recommended tool_profile_name**: `write_only`
- **use_todo_middleware**: `false` (writing is a focused output operation)
- **System prompt guidance**: Instruct to read synthesis and evidence files, produce a polished report with sections, citations, and clear structure, write to `/outputs/report.md`.

## domain_specialist
- **Purpose**: Deep-dives into a specific biotech subdomain (e.g., CRISPR mechanisms, clinical trial design, protein folding, drug delivery systems). Combines search with domain expertise.
- **Recommended tool_profile_name**: `default_research`
- **use_todo_middleware**: `true` (domain deep-dives are inherently multi-step)
- **System prompt guidance**: Instruct with specific domain context, key terminology, what to look for, and to write a domain-specific analysis to `/outputs/domain_analysis.md`.

## Selection Guidelines

- Assign **1-3 subagents** per TaskDef based on task complexity and stage.
- Prefer `source_finder` for the first task in a dependency chain.
- Prefer `writer` or `synthesizer` for terminal tasks.
- For mid-chain analysis tasks, combine `evidence_extractor` + `verifier`.
- Only assign `use_todo_middleware: true` for subagents that will perform multi-step, iterative work.
- Write **distinct, specific** system prompts per subagent — never use generic "you are a researcher" prompts.
- Include in each main agent system prompt explicit instructions on **when** to delegate to each subagent.
