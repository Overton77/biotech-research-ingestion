"""
Scoring rubrics and threshold constants for research mission evaluation.

These rubrics are used by both LLM-as-judge evaluators and code-based evaluators
to maintain consistent quality standards across experiments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Score thresholds (normalized 0.0 – 1.0)
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 0.7
EXCELLENT_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Report completeness rubric (used by LLM-as-judge)
# ---------------------------------------------------------------------------

REPORT_COMPLETENESS_RUBRIC = """
You are evaluating a biotech research report for completeness.

Score the report on a scale of 0 to 10 based on these criteria:

1. **Section Coverage (0-3):** Does the report include all required sections?
   Check against the required_sections list provided. Each missing section
   costs 1 point. All present = 3.

2. **Depth of Coverage (0-3):** Does each section contain substantive content?
   - 0 = sections are empty or have only headers
   - 1 = superficial (1-2 sentences per section)
   - 2 = moderate (key facts covered, some detail)
   - 3 = thorough (specific data points, examples, nuance)

3. **Source Attribution (0-2):** Are claims backed by cited sources?
   - 0 = no sources cited
   - 1 = some sources, but many claims unsupported
   - 2 = most claims have source attribution

4. **Actionability (0-2):** Does the report provide actionable intelligence?
   - 0 = purely descriptive, no insights
   - 1 = some insights but vague
   - 2 = clear findings with specific next steps or recommendations

Return a JSON object with:
- total_score: integer 0-10
- section_coverage: integer 0-3
- depth: integer 0-3
- source_attribution: integer 0-2
- actionability: integer 0-2
- reasoning: string explaining the scores
- missing_sections: list of section names that are absent or empty
""".strip()

# ---------------------------------------------------------------------------
# Report accuracy rubric (used by LLM-as-judge)
# ---------------------------------------------------------------------------

REPORT_ACCURACY_RUBRIC = """
You are evaluating a biotech research report for factual accuracy.

Score the report on a scale of 0 to 10:

1. **Factual Consistency (0-4):** Are the stated facts internally consistent?
   Do numbers, names, dates, and claims agree with each other throughout?

2. **Source Alignment (0-3):** Where sources are cited, does the report
   accurately represent what those sources say? Look for misquotations,
   exaggerations, or misattributions.

3. **Claim Calibration (0-3):** Are uncertainty levels appropriate?
   Does the report distinguish between confirmed facts, likely inferences,
   and speculative claims? Overconfident claims about uncertain data lose points.

Return a JSON object with:
- total_score: integer 0-10
- factual_consistency: integer 0-4
- source_alignment: integer 0-3
- claim_calibration: integer 0-3
- reasoning: string explaining the scores
- flagged_claims: list of specific claims that appear inaccurate or unsupported
""".strip()

# ---------------------------------------------------------------------------
# Iteration convergence thresholds
# ---------------------------------------------------------------------------

MIN_CONFIDENCE_IMPROVEMENT_PER_ITERATION = 0.05
DIMINISHING_RETURNS_THRESHOLD = 0.02
