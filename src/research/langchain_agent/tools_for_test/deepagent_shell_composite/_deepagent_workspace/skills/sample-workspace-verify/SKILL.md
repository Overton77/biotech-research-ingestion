---
name: sample-workspace-verify
description: Use this skill when the user asks for a skill verification test, sample workspace verification, or explicitly mentions verifying the deep agent shell workspace.
---

# Sample workspace verification

When this skill applies, perform these steps in order:

1. Use the **grep** tool (not shell `grep`) with pattern `GREP_MARKER_XYZZY`. Search under path `/data` or use glob `*.txt` scoped to `/data`.
2. Use **read_file** on `/data/sample.txt` and confirm the marker appears in the file.
3. Use **write_file** to create `/skill_proof.txt` whose entire content is exactly one line: `SKILL_PROOF_OK` (no extra whitespace or blank lines after it).
4. In your final assistant message, include the exact token `SKILL_PROOF_OK` and state clearly that you used the **grep** tool as part of the workflow.
