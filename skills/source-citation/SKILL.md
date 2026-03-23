---
name: source-citation
description: >
  How to cite sources inline, build reference lists, and maintain source
  provenance during biotech research. Use this skill whenever you conduct
  web searches or reference external data. Ensures every claim in your
  output can be traced back to a URL or document.
---

# Source Citation

## Why it matters

Biotech research reports are read by regulatory, clinical, and investment
audiences. Every factual claim must be traceable. Source tracking happens
automatically at the tool level — your job is to USE the index that is built
for you and to write citations inline.

## How source tracking works

Every time you or a subagent calls a search tool (tavily_search,
tavily_extract, tavily_map, tavily_crawl), the result is automatically logged
to `sources/search_results.jsonl` in the task workspace. After the task
completes, a deduplicated `sources/index.json` is produced.

You do not need to manually log sources. You DO need to reference them.

## Inline citation format

When making a factual claim, append `[src:N]` where N is the 1-based index
of the source in `sources/index.json`. If you have not yet checked the
source index, retrieve it with a file read before writing your final report.

Example:
> Pembrolizumab received FDA approval for MSI-H cancers in 2017 [src:3].

## Reference section format

End every report with a `## References` section:

```
## References

[1] Title or URL — accessed YYYY-MM-DD
[2] ...
```

Use the URL as the identifier when no title is available.

## Output file conventions

- Write your final synthesis report to `outputs/<task_name>.md`
- Source index is at `sources/index.json` (read-only after task completion)
- Do NOT copy sources into your report body — cite by index only

## What to do when sources are unavailable

If a claim is derived from the LLM's training knowledge rather than a live
search, use `[knowledge]` instead of `[src:N]` and note the limitation in
the report's caveats section.

## Shared Mission Knowledge

A shared namespace `/shared/` is available for cross-task collaboration:

- `/shared/entities/companies.jsonl` — write discovered company names here
- `/shared/entities/drugs.jsonl` — write discovered drug/compound names
- `/shared/sources/master_index.jsonl` — aggregated source index

At task start, read `/shared/` to get context from prior tasks.
Write important discoveries to `/shared/` for subsequent tasks.
