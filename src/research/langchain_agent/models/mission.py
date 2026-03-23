"""
Research mission and stage models. MissionStage supports dependencies;
when a stage depends on another, it receives that stage's final_report as input.

Stages may optionally be iterative: when iterative_config is set on a MissionStage,
the runner executes the stage in a loop — each iteration produces a report and
structured next-steps, and the runner decides whether to continue.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.research.langchain_agent.agent.config import MissionSliceInput, ResearchPromptSpec


# -----------------------------------------------------------------------------
# Iterative stage configuration
# -----------------------------------------------------------------------------


class IterativeStageConfig(BaseModel):
    """
    Configuration for stage-level iteration.

    When attached to a MissionStage, the DAG runner will execute that stage
    in a loop rather than as a single pass.  Each iteration produces a report
    and a NextStepsArtifact; the runner evaluates stop conditions between
    iterations and carries forward context.
    """

    max_iterations: int = 3
    completion_criteria: str = ""
    carry_forward_reports: int = 2
    stop_on_no_next_steps: bool = True

QUALIA_BASE_DOMAIN = "https://www.qualialife.com"
QUALIA_MISSION_ID = "mission-qualia-life-sciences-001"

ELYSIUM_BASE_DOMAIN = "https://www.elysiumhealth.com"
ELYSIUM_MISSION_ID = "mission-elysium-health-001"


# -----------------------------------------------------------------------------
# Stage-specific prompt specs (Qualia 3-stage)
# -----------------------------------------------------------------------------


def _fundamentals_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a corporate research agent focused on company fundamentals "
            "for Qualia Life Sciences."
        ),
        domain_scope=[
            "Qualia Life Sciences LLC / Qualia Life Sciences",
            "company history, founding, and evolution",
            "recent news and stories (last 12–24 months)",
            "corporate structure, location, headquarters",
            "official domain: qualialife.com",
        ],
        workflow=[
            "Start with search_web (include qualialife.com) to find company overview and recent news.",
            "Use map_website on qualialife.com to find About, Company, Press, or News pages.",
            "Use extract_from_urls on key pages for company history, recent stories, and fundamentals.",
            "Save intermediate findings to the filesystem.",
            "Write a final report synthesizing company fundamentals.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['qualialife.com'] and topic='news' for recent stories.",
            "Use map_website on qualialife.com to discover About/Company/Press/News URLs.",
            "Use extract_from_urls for high-value pages.",
            "Use the task tool with browser_control subagent if pages need scrolling or dynamic content.",
        ],
        subagent_guidance=[
            "Call browser_control when pages need scrolling, clicking, or dynamic content.",
            "Provide clear instructions: URL(s) and what to extract.",
        ],
        practical_limits=[
            "Focus on qualialife.com and reputable business sources.",
            "Avoid redundant searches.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_discovery_plan.md",
            "runs/<task_slug>/02_company_overview.md",
            "runs/<task_slug>/03_recent_stories.md",
            "runs/<task_slug>/04_draft_report.md",
        ],
    )


def _products_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a supplement product research agent focused on cataloging "
            "Qualia Life Sciences products."
        ),
        domain_scope=[
            "Qualia Life Sciences supplement products",
            "product names, ingredients, dosages",
            "prices and purchasing options",
            "product specifications and benefits",
        ],
        workflow=[
            "Use search_web and map_website to find product catalog URLs on qualialife.com.",
            "Use extract_from_urls on static product pages.",
            "Use the task tool + browser_control subagent when pages need scrolling/clicking for prices and specs.",
            "Save intermediate product data; write a final product catalog report.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['qualialife.com'].",
            "Use map_website to discover product URLs.",
            "Use the task tool with browser_control when product pages need interaction.",
        ],
        subagent_guidance=[
            "Call browser_control when product pages require scrolling, clicking, or dynamic content.",
            "Provide URL(s) and what to extract (price, ingredients, specs).",
        ],
        practical_limits=[
            "Focus on product-related paths on qualialife.com.",
            "Batch extractions.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_product_discovery_plan.md",
            "runs/<task_slug>/02_qualia_product_urls.md",
            "runs/<task_slug>/03_product_details_raw.md",
            "runs/<task_slug>/04_product_specs_prices_ingredients.md",
            "runs/<task_slug>/05_draft_product_catalog.md",
        ],
    )


def _leadership_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a corporate research agent focused on leadership and "
            "scientific advisors of Qualia Life Sciences."
        ),
        domain_scope=[
            "Qualia Life Sciences leadership team",
            "executives, founders, C-suite",
            "scientific advisors, medical advisors",
            "team page, about page, science team",
        ],
        workflow=[
            "Use search_web and map_website to find leadership/team/about pages on qualialife.com.",
            "Use extract_from_urls to capture names, titles, bios.",
            "Use browser_control subagent if pages need interaction.",
            "Save intermediate findings; write a final leadership report.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['qualialife.com'].",
            "Look for Team, About, Science Team, Leadership, Advisors.",
            "Use browser_control when pages need scrolling or dynamic content.",
        ],
        subagent_guidance=[
            "Call browser_control when team/leadership pages require interaction.",
            "Provide URL(s) and what to extract (names, titles, bios).",
        ],
        practical_limits=[
            "Focus on qualialife.com team and advisor pages.",
            "Capture all named leaders and advisors.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_leadership_discovery.md",
            "runs/<task_slug>/02_team_and_advisors_raw.md",
            "runs/<task_slug>/03_draft_leadership_report.md",
        ],
    )


# -----------------------------------------------------------------------------
# Mission stage: slice input + prompt spec + dependencies
# -----------------------------------------------------------------------------


class MissionStage(BaseModel):
    """One stage of a research mission: slice input + prompt spec + optional dependencies.

    When ``iterative_config`` is set the DAG runner will execute this stage
    in a bounded iteration loop instead of a single pass.
    """

    slice_input: MissionSliceInput
    prompt_spec: ResearchPromptSpec
    execution_reminders: list[str] = Field(
        default_factory=lambda: [
            "Use runs/, reports/, and scratch/ as main folders.",
            "Save intermediate data; write the final report to reports/.",
            "Use recalled memories as hints, not unquestioned truth.",
        ]
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="task_slug of stages that must complete before this one; this stage receives their final_report.",
    )
    iterative_config: IterativeStageConfig | None = Field(
        default=None,
        description="When set, the stage runs iteratively (multiple bounded passes) instead of a single pass.",
    )


# -----------------------------------------------------------------------------
# Research mission
# -----------------------------------------------------------------------------


class ResearchMission(BaseModel):
    """Multi-stage research mission (e.g. Qualia Life Sciences 3-stage)."""

    mission_id: str = QUALIA_MISSION_ID
    mission_name: str = "Qualia Life Sciences Full Entity Research"
    base_domain: str = QUALIA_BASE_DOMAIN
    stages: list[MissionStage] = Field(default_factory=list)

    @classmethod
    def qualia_life_sciences_3_stage(cls) -> "ResearchMission":
        """Build the hardcoded 3-stage Qualia mission."""
        return cls(
            mission_id=QUALIA_MISSION_ID,
            mission_name="Qualia Life Sciences Full Entity Research",
            base_domain=QUALIA_BASE_DOMAIN,
            stages=[
                MissionStage(
                    slice_input=MissionSliceInput(
                        task_id="qualia-fundamentals-001",
                        mission_id=QUALIA_MISSION_ID,
                        task_slug="qualia-company-fundamentals",
                        user_objective=(
                            "Research fundamental company information for Qualia Life Sciences: "
                            "company history, founding, location, structure, and recent news/stories "
                            "(last 12–24 months). Use qualialife.com as primary source. "
                            "Save intermediate findings to runs/ and scratch/; write final report to reports/."
                        ),
                        targets=["Qualia Life Sciences", "qualialife.com"],
                        selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                        stage_type="entity_validation",
                        max_step_budget=10,
                    ),
                    prompt_spec=_fundamentals_prompt_spec(),
                    execution_reminders=[
                        "Focus on company history, structure, and recent stories.",
                        "Use runs/, reports/, and scratch/. Save intermediate findings.",
                    ],
                    dependencies=[],
                ),
                MissionStage(
                    slice_input=MissionSliceInput(
                        task_id="qualia-products-001",
                        mission_id=QUALIA_MISSION_ID,
                        task_slug="qualia-products-and-specs",
                        user_objective=(
                            "Catalog all supplement products from Qualia Life Sciences (qualialife.com). "
                            "For each product: name, price, ingredients, dosages, and specifications. "
                            "Use search_web and map_website to find product URLs. Use extract_from_urls for static pages. "
                            "When product pages need browser interaction (dynamic content, scrolling), "
                            "use the task tool with browser_control subagent. "
                            "Save product data to runs/ and scratch/; write final catalog to reports/."
                        ),
                        targets=["Qualia Life Sciences", "qualialife.com"],
                        selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                        stage_type="targeted_extraction",
                        max_step_budget=12,
                    ),
                    prompt_spec=_products_prompt_spec(),
                    execution_reminders=[
                        "Catalog all products with prices, ingredients, specs.",
                        "Use browser_control subagent when product pages need scrolling/clicking.",
                        "Save to runs/ and scratch/; final report to reports/.",
                    ],
                    dependencies=["qualia-company-fundamentals"],
                ),
                MissionStage(
                    slice_input=MissionSliceInput(
                        task_id="qualia-leadership-001",
                        mission_id=QUALIA_MISSION_ID,
                        task_slug="qualia-leadership-and-advisors",
                        user_objective=(
                            "Research all leadership and scientific advisors of Qualia Life Sciences. "
                            "Include executives, founders, C-suite, medical/scientific advisors. "
                            "For each person: name, title/role, brief bio. "
                            "Use qualialife.com (Team, About, Science Team pages) as primary source. "
                            "Save intermediate findings; write final leadership report to reports/."
                        ),
                        targets=["Qualia Life Sciences", "qualialife.com"],
                        selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                        stage_type="targeted_extraction",
                        max_step_budget=10,
                    ),
                    prompt_spec=_leadership_prompt_spec(),
                    execution_reminders=[
                        "Capture all leadership and scientific advisors with names, titles, bios.",
                        "Use runs/, reports/, scratch/. Write final report to reports/.",
                    ],
                    dependencies=[],
                ),
            ],
        )


# -----------------------------------------------------------------------------
# Elysium Health: stage-specific prompt specs (3-stage)
# -----------------------------------------------------------------------------


def _elysium_fundamentals_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a corporate research agent focused on company fundamentals "
            "for Elysium Health (healthy aging / longevity biotech supplements)."
        ),
        domain_scope=[
            "Elysium Health",
            "company history, founding, mission (translate aging research into products)",
            "recent news and stories (last 12–24 months)",
            "corporate structure, location, headquarters",
            "official domain: elysiumhealth.com",
            "partnerships (e.g. Oxford, Yale), clinical trials, scientific validation",
        ],
        workflow=[
            "Start with search_web (include elysiumhealth.com) to find company overview and recent news.",
            "Use map_website on elysiumhealth.com to find About, Company, Science, Press, or News pages.",
            "Use extract_from_urls on key pages for company history, recent stories, and fundamentals.",
            "Use the task tool with browser_control subagent if pages need scrolling or dynamic content.",
            "Save intermediate findings to the filesystem.",
            "Write a final report synthesizing company fundamentals.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['elysiumhealth.com'] and topic='news' for recent stories.",
            "Use map_website on elysiumhealth.com to discover About/Company/Science/Press URLs.",
            "Use extract_from_urls for high-value pages.",
            "Use the task tool with browser_control subagent if pages need scrolling or dynamic content.",
        ],
        subagent_guidance=[
            "Call browser_control when pages need scrolling, clicking, or dynamic content.",
            "Provide clear instructions: URL(s) and what to extract.",
        ],
        practical_limits=[
            "Focus on elysiumhealth.com and reputable business/science sources.",
            "Avoid redundant searches.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_discovery_plan.md",
            "runs/<task_slug>/02_company_overview.md",
            "runs/<task_slug>/03_recent_stories.md",
            "runs/<task_slug>/04_draft_report.md",
        ],
    )


def _elysium_products_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a supplement product research agent focused on cataloging "
            "Elysium Health products (Basis, Matter, Signal, Mosaic, Format, Vision, Senolytic, Cofactor, etc.)."
        ),
        domain_scope=[
            "Elysium Health supplement products",
            "product names, ingredients, dosages, NAD+ and longevity science",
            "prices and purchasing options",
            "product specifications, clinical evidence, and benefits",
        ],
        workflow=[
            "Use search_web and map_website to find product catalog and product pages on elysiumhealth.com.",
            "Use extract_from_urls on static product pages for ingredients, descriptions, and specs where possible.",
            "When product pages need scrolling, clicking, or dynamic content to reveal full specs or prices, use the task tool to call the browser_control subagent (Playwright). Give it a clear task: e.g. 'Go to [URL], scroll to reveal full ingredient list and price, extract product specs.'",
            "Save intermediate product data; write a final product catalog report.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['elysiumhealth.com'].",
            "Use map_website to discover product URLs (e.g. /basis, /matter, /collections/supplements).",
            "Use the task tool with browser_control subagent when product pages need interaction for full specs or prices.",
            "In the task tool input, be specific: URL(s), what to extract (price, ingredients, dosage, clinical claims).",
        ],
        subagent_guidance=[
            "You have access to a browser_control subagent via the task tool (Playwright). Use it when product pages on elysiumhealth.com require scrolling, clicking, or interaction to reveal prices, full ingredient lists, or specs.",
            "Call browser_control when extract_from_urls returns incomplete data or pages are dynamic.",
            "Provide URL(s) and what to extract (price, ingredients, dosage, product specs).",
        ],
        practical_limits=[
            "Focus on product-related paths on elysiumhealth.com.",
            "Batch extractions; use browser_control when needed for full product specs.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_product_discovery_plan.md",
            "runs/<task_slug>/02_elysium_product_urls.md",
            "runs/<task_slug>/03_product_details_raw.md",
            "runs/<task_slug>/04_product_specs_prices_ingredients.md",
            "runs/<task_slug>/05_draft_product_catalog.md",
        ],
    )


def _elysium_leadership_prompt_spec() -> ResearchPromptSpec:
    return ResearchPromptSpec(
        agent_identity=(
            "You are a corporate research agent focused on leadership and "
            "scientific advisors of Elysium Health."
        ),
        domain_scope=[
            "Elysium Health leadership team",
            "Dr. Leonard Guarente (chief scientist, MIT, SIR2/NAD+ research)",
            "executives, founders, C-suite",
            "scientific advisors (25+), medical advisors",
            "team page, about page, science team",
        ],
        workflow=[
            "You will receive the company fundamentals report from a prior stage; use it for context.",
            "Use search_web and map_website to find leadership/team/about/science pages on elysiumhealth.com.",
            "Use extract_from_urls to capture names, titles, bios.",
            "Use browser_control subagent if pages need interaction.",
            "Save intermediate findings; write a final leadership report.",
        ],
        tool_guidance=[
            "Use search_web with include_domains=['elysiumhealth.com'].",
            "Look for Team, About, Science, Leadership, Advisors, Our Scientists.",
            "Use browser_control when pages need scrolling or dynamic content.",
        ],
        subagent_guidance=[
            "Call browser_control when team/leadership pages require interaction.",
            "Provide URL(s) and what to extract (names, titles, bios).",
        ],
        practical_limits=[
            "Focus on elysiumhealth.com team and advisor pages.",
            "Capture all named leaders and scientific advisors.",
        ],
        filesystem_rules=[
            "Use runs/, reports/, and scratch/ as main folders.",
            "Write relative paths only.",
        ],
        intermediate_files=[
            "runs/<task_slug>/01_leadership_discovery.md",
            "runs/<task_slug>/02_team_and_advisors_raw.md",
            "runs/<task_slug>/03_draft_leadership_report.md",
        ],
    )


# -----------------------------------------------------------------------------
# ResearchMission: Elysium Health 3-stage
# -----------------------------------------------------------------------------


def elysium_health_3_stage_mission() -> ResearchMission:
    """Build the 3-stage Elysium Health mission. Leadership depends on company fundamentals (receives that report)."""
    return ResearchMission(
        mission_id=ELYSIUM_MISSION_ID,
        mission_name="Elysium Health Full Entity Research",
        base_domain=ELYSIUM_BASE_DOMAIN,
        stages=[
            MissionStage(
                slice_input=MissionSliceInput(
                    task_id="elysium-fundamentals-001",
                    mission_id=ELYSIUM_MISSION_ID,
                    task_slug="elysium-company-fundamentals",
                    user_objective=(
                        "Research fundamental company information for Elysium Health: "
                        "company history, founding, mission (translating aging research into products), "
                        "location, structure, recent news/stories (last 12–24 months), "
                        "and key partnerships (e.g. Oxford, Yale) and clinical validation. "
                        "Use elysiumhealth.com as primary source. "
                        "Save intermediate findings to runs/ and scratch/; write final report to reports/."
                    ),
                    targets=["Elysium Health", "elysiumhealth.com"],
                    selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                    stage_type="entity_validation",
                    max_step_budget=10,
                ),
                prompt_spec=_elysium_fundamentals_prompt_spec(),
                execution_reminders=[
                    "Focus on company history, structure, mission, and recent stories.",
                    "Use runs/, reports/, and scratch/. Save intermediate findings.",
                ],
                dependencies=[],
            ),
            MissionStage(
                slice_input=MissionSliceInput(
                    task_id="elysium-products-001",
                    mission_id=ELYSIUM_MISSION_ID,
                    task_slug="elysium-products-and-specs",
                    user_objective=(
                        "Catalog all supplement products from Elysium Health (elysiumhealth.com). "
                        "Include Basis, Matter, Signal, Mosaic, Format, Vision, Senolytic Complex, Cofactor, and any others. "
                        "For each product: name, price, ingredients, dosages, and specifications. "
                        "Use search_web and map_website to find product URLs. Use extract_from_urls for static pages. "
                        "When product pages need browser interaction (dynamic content, scrolling to see full specs or prices), "
                        "use the task tool with browser_control subagent (Playwright) and give clear instructions. "
                        "Save product data to runs/ and scratch/; write final catalog to reports/."
                    ),
                    targets=["Elysium Health", "elysiumhealth.com"],
                    selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                    stage_type="targeted_extraction",
                    max_step_budget=12,
                ),
                prompt_spec=_elysium_products_prompt_spec(),
                execution_reminders=[
                    "Catalog all products with prices, ingredients, specs. Use browser_control (Playwright) subagent when product pages need scrolling/clicking for full specs.",
                    "Save to runs/ and scratch/; final report to reports/.",
                ],
                dependencies=[],
            ),
            MissionStage(
                slice_input=MissionSliceInput(
                    task_id="elysium-leadership-001",
                    mission_id=ELYSIUM_MISSION_ID,
                    task_slug="elysium-leadership-and-advisors",
                    user_objective=(
                        "Research all leadership and scientific advisors of Elysium Health. "
                        "You will receive the company fundamentals report from a prior stage—use it for context. "
                        "Include executives, founders, C-suite, and scientific advisors (e.g. Dr. Leonard Guarente). "
                        "For each person: name, title/role, brief bio. "
                        "Use elysiumhealth.com (Team, About, Science pages) as primary source. "
                        "Save intermediate findings; write final leadership report to reports/."
                    ),
                    targets=["Elysium Health", "elysiumhealth.com"],
                    selected_tool_names=["search_web", "map_website", "extract_from_urls"],
                    stage_type="targeted_extraction",
                    max_step_budget=10,
                ),
                prompt_spec=_elysium_leadership_prompt_spec(),
                execution_reminders=[
                    "Use the company fundamentals report provided as context. Capture all leadership and scientific advisors with names, titles, bios.",
                    "Use runs/, reports/, scratch/. Write final report to reports/.",
                ],
                dependencies=["elysium-company-fundamentals"],
            ),
        ],
    )


# Pre-built instances for tests and CLI
QUALIA_RESEARCH_MISSION = ResearchMission.qualia_life_sciences_3_stage()
ELYSIUM_RESEARCH_MISSION = elysium_health_3_stage_mission()
