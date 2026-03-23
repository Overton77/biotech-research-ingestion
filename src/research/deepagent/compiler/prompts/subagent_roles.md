# Subagent Design Guidance

Design subagents appropriate to each task. **Role types and names are not fixed** — use whatever specialization best serves the task. Do not select from a fixed catalogue; instead, define 1–3 subagents per task with distinct roles and clear deliverables.

## Design principles

- **One clear role per subagent**: Each subagent should have a single, well-defined responsibility (e.g. find sources, extract evidence, synthesize, write a report).
- **Expected output**: For each subagent, specify what it should produce and where. Use **expected_output_format** (human-readable description of content/format) and **expected_output_path** (path relative to the subagent’s workspace, e.g. `outputs/sources.md`). The main agent and runtime use these to know what to expect and where to find results.
- **Tool profile**: Choose `search_only`, `write_only`, or `default_research` based on whether the subagent mainly searches the web, mainly writes/reads files, or does both.
- **Todo middleware**: Set `use_todo_middleware: true` only for subagents that perform multi-step, iterative work (e.g. discovering many sources, iterating over documents). Use `false` for focused single-pass operations.
- **System prompt**: Write a **distinct, specific** system prompt per subagent. Include where to write the main deliverable (matching **expected_output_path**). Never use generic “you are a researcher” prompts.
- **Main agent delegation**: In the main_agent system_prompt, include explicit instructions on **when** to delegate to each subagent by name.

## Example 1: “Find all products of a company”

For a task that requires discovering and listing a company’s products:

- **Subagent role**: Product discoverer (or similar name you choose).
- **expected_output_format**: “Markdown table of products: name, category, description, link.”
- **expected_output_path**: `outputs/company_products.md`
- **tool_profile_name**: `search_only` (web search to find product pages).
- **use_todo_middleware**: `true` (multi-step discovery).
- **System prompt (summary)**: Instruct to search for the company’s product pages, extract product name, category, description, and link, deduplicate, and write the table to `outputs/company_products.md`.

## Example 2: “Structured list of sources with title, URL, relevance”

For a task that requires gathering and ranking sources:

- **Subagent role**: Source finder (or similar name).
- **expected_output_format**: “Structured list of sources with title, URL, relevance score, and one-line summary.”
- **expected_output_path**: `outputs/sources.md`
- **tool_profile_name**: `search_only`
- **use_todo_middleware**: `true`
- **System prompt (summary)**: Instruct to find authoritative sources (e.g. .gov, .edu, peer-reviewed), rank by relevance, and write the list to `outputs/sources.md`.

Use these examples only as inspiration. Design 1–3 subagents per task with distinct roles; names and specializations are up to you.
