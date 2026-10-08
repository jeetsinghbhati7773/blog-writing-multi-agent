from __future__ import annotations

import operator
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from pydantic import BaseModel, Field


from enum import Enum


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class VisualType(str, Enum):
    TECHNICAL_DIAGRAM = "TECHNICAL_DIAGRAM"
    ARCHITECTURE_DIAGRAM = "ARCHITECTURE_DIAGRAM"
    PROCESS_FLOW = "PROCESS_FLOW"
    INFOGRAPHIC = "INFOGRAPHIC"
    CONCEPTUAL_ILLUSTRATION = "CONCEPTUAL_ILLUSTRATION"
    COMPARISON_GRAPHIC = "COMPARISON_GRAPHIC"
    CODE_VISUALIZATION = "CODE_VISUALIZATION"
    HERO_IMAGE = "HERO_IMAGE"


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    section_title: str = Field(..., description="Title of the blog section (H2) where this image belongs.")
    visual_type: Literal[
        "TECHNICAL_DIAGRAM",
        "ARCHITECTURE_DIAGRAM",
        "PROCESS_FLOW",
        "INFOGRAPHIC",
        "CONCEPTUAL_ILLUSTRATION",
        "COMPARISON_GRAPHIC",
        "CODE_VISUALIZATION",
        "HERO_IMAGE"
    ] = "TECHNICAL_DIAGRAM"
    subject: str = Field(..., description="Core subject / theme to be illustrated.")
    composition: str = Field(..., description="Layout & spatial arrangement instruction (e.g. Left-to-right flow).")
    visual_style: str = Field("Clean technical editorial illustration", description="Artistic/technical rendering style.")
    color_palette: str = Field("Modern dark blue and teal accent palette", description="Color direction.")
    aspect_ratio: Literal["16:9", "1:1", "4:3"] = "16:9"
    avoid: List[str] = Field(default_factory=lambda: ["generic AI brains", "circuit board patterns", "excessive text", "watermarks"], description="Negative constraints.")
    alt: str
    caption: str
    prompt: str = Field(..., description="Structured prompt sent to the visual model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "high"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: Optional[str] = Field(None, description="Modified markdown text with placeholders.")
    images: List[ImageSpec] = Field(default_factory=list)
    visual_diversity_notes: Optional[str] = Field(None, description="Notes ensuring no visual type repetition.")


class SourceFilterScore(BaseModel):
    url: str
    relevance_score: float = Field(..., description="Relevance to topic (0.0 to 1.0)")
    authority_score: float = Field(..., description="Source authority/credibility (0.0 to 1.0)")
    freshness_score: float = Field(..., description="Freshness & recency alignment (0.0 to 1.0)")
    is_duplicate: bool = False
    keep: bool = Field(..., description="True if source passes quality bar")
    reason: str


class SourceFilterResult(BaseModel):
    scores: List[SourceFilterScore] = Field(default_factory=list)


class ContradictionItem(BaseModel):
    topic_claim: str
    source_a_url: str
    source_a_claim: str
    source_b_url: str
    source_b_claim: str
    resolution_guidance: str


class ContradictionReport(BaseModel):
    contradictions: List[ContradictionItem] = Field(default_factory=list)


class ResearchResult(BaseModel):
    topic: str
    key_facts: List[str] = Field(default_factory=list)
    technical_details: List[str] = Field(default_factory=list)
    code_examples: List[str] = Field(default_factory=list)
    sources: List[EvidenceItem] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    filtered_sources_count: int = 0


class OutlineValidationWarning(BaseModel):
    category: Literal["task_count", "intro_missing", "conclusion_missing", "word_distribution", "duplicate_topic"]
    severity: Literal["info", "warning", "error"]
    message: str


class OutlineValidationResult(BaseModel):
    passed: bool
    warnings: List[OutlineValidationWarning] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    claim_id: str
    claim: str
    classification: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    confidence: float = Field(default=1.0, description="Classification confidence score from 0.0 to 1.0")
    supporting_sources: List[str] = Field(default_factory=list)
    contradicting_sources: List[str] = Field(default_factory=list)
    supported_components: List[str] = Field(default_factory=list)
    unsupported_components: List[str] = Field(default_factory=list)
    explanation: str

    @property
    def verdict(self) -> str:
        if self.classification == "SUPPORTED":
            return "verified"
        elif self.classification == "CONTRADICTED":
            return "contradicted"
        elif self.classification == "PARTIALLY_SUPPORTED":
            return "partially_supported"
        return "unsupported"

    @property
    def source_url(self) -> Optional[str]:
        if self.supporting_sources:
            return self.supporting_sources[0]
        if self.contradicting_sources:
            return self.contradicting_sources[0]
        return None


# Backward compatibility alias
FactCheckItem = ClaimVerification


class FactCheckReport(BaseModel):
    score: float = Field(..., description="Fact-check grounding score from 0.0 to 10.0")
    total_claims_checked: int = 0
    verified_claims_count: int = 0
    claims: List[ClaimVerification] = Field(default_factory=list)
    critical_errors: List[str] = Field(default_factory=list)
    recommendation: Literal["PASS", "REVISE", "HUMAN_REVIEW"] = "PASS"


class SEOFAQItem(BaseModel):
    question: str
    answer: str


class SEOPlan(BaseModel):
    meta_title: str
    meta_description: str
    primary_keyword: str
    secondary_keywords: List[str] = Field(default_factory=list)
    suggested_slug: str
    target_readability_level: str
    faq_items: List[SEOFAQItem] = Field(default_factory=list)
    seo_score: float = Field(..., description="SEO optimization score from 0.0 to 10.0")


class ArticleStyleProfile(str, Enum):
    TECHNICAL_TUTORIAL = "Technical Tutorial"
    BEGINNER_FRIENDLY = "Beginner Friendly"
    RESEARCH_STYLE = "Research Style"
    DEVELOPER_BLOG = "Developer Blog"
    LINKEDIN_POST = "LinkedIn Post"
    SEO_BLOG = "SEO Blog"


class CriticIssue(BaseModel):
    category: Literal["consistency", "relevance", "completeness", "readability", "repetition", "structure", "grounding"]
    severity: Literal["low", "medium", "high"]
    description: str
    suggestion: str


class CriticEvaluation(BaseModel):
    score: float = Field(..., description="Overall article quality score from 0.0 to 10.0")
    passed: bool = Field(..., description="True if overall quality score is >= 7.5 and no critical issues exist")
    summary: str = Field(..., description="High-level evaluation summary of the draft")
    issues: List[CriticIssue] = Field(default_factory=list, description="Specific identified issues")
    revision_instructions: Optional[str] = Field(None, description="Actionable revision prompt if revision is needed")


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REGENERATE = "regenerate"


class SocialPosts(BaseModel):
    twitter_thread: List[str] = Field(
        ...,
        description="List of 5-7 tweets, max 280 chars per tweet, including hooks and hashtags."
    )
    linkedin_post: str = Field(
        ...,
        description="Formatted with bold text, emojis, short paragraphs, call-to-action."
    )
    newsletter_summary: str = Field(
        ...,
        description="Email digest style, 200-300 words."
    )


class State(TypedDict):
    topic: str
    audience: Optional[str]
    tone: Optional[str]
    target_length: Optional[int]
    keywords: Optional[List[str]]
    instructions: Optional[str]

    # Style Profile
    style_profile: Optional[str]

    # routing / research / filter / contradiction
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    source_filter_scores: List[dict]
    research_result: Optional[ResearchResult]
    contradictions: List[dict]

    # plan & validation
    plan: Optional[Plan]
    plan_version: int
    outline_validation: Optional[OutlineValidationResult]

    # HITL approval #1 (Plan)
    approval_status: ApprovalStatus
    human_approval: Optional[dict]
    human_feedback: Optional[str]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer / fact check / critic / revision / SEO
    merged_md: str
    fact_check_report: Optional[FactCheckReport]
    critic_evaluation: Optional[CriticEvaluation]
    critic_retry_count: int
    max_critic_retries: int
    seo_plan: Optional[SEOPlan]

    # images & placeholders
    md_with_placeholders: str
    image_specs: List[dict]

    # HITL approval #2 (Final Content)
    final_approval_status: Optional[ApprovalStatus]
    final_human_feedback: Optional[str]

    # social media syndication
    social_posts: Optional[SocialPosts]

    # observability & metrics
    metrics: Optional[Dict[str, Any]]

    final: str
    errors: Optional[List[str]]
    retry_count: int


OverallState = State







