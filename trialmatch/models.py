"""Data shapes shared by the pipeline, the API and the evaluation scripts.

Two families live here:

* ``LLM*`` models are the structured-output schemas the model must fill in.
  They are kept flat and free of numeric/length constraints so they translate
  cleanly to a strict JSON schema.
* Everything else is what the program stores and serves after it has verified
  and post-processed the model's answer.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CriterionType = Literal["inclusion", "exclusion"]
# "met" means the patient satisfies the criterion AS WRITTEN. For an exclusion
# criterion that is bad news for enrollment; see ``effect_of``.
Verdict = Literal["met", "not_met", "unknown"]
Confidence = Literal["high", "medium", "low"]
Effect = Literal["supports", "blocks", "unresolved"]
Overall = Literal["possibly_eligible", "not_eligible", "needs_more_info"]

VERDICT_LABELS = {"met": "Met", "not_met": "Not met", "unknown": "Insufficient information"}
OVERALL_LABELS = {
    "possibly_eligible": "Possibly eligible",
    "not_eligible": "Likely not eligible",
    "needs_more_info": "More information needed",
}


def effect_of(criterion_type: str, verdict: str) -> Effect:
    """Translate a verdict into what it means for enrollment.

    Exclusion logic is inverted: a patient who *meets* an exclusion criterion
    cannot be enrolled, and one who does not meet it clears that criterion.
    """
    if verdict == "unknown":
        return "unresolved"
    if criterion_type == "inclusion":
        return "supports" if verdict == "met" else "blocks"
    return "blocks" if verdict == "met" else "supports"


# --------------------------------------------------------------------------
# Structured-output schemas (filled in by the model)
# --------------------------------------------------------------------------


class Concept(BaseModel):
    original: str = Field(description="The term exactly as written in the criterion.")
    normalized: Optional[str] = Field(
        description="Standard preferred name (e.g. generic drug name, standard disease or lab "
        "test name). null when there is no confident standard name; never force a match."
    )
    category: Literal["condition", "drug", "lab_test", "procedure", "other"]


class Threshold(BaseModel):
    parameter: str = Field(description="What is measured, e.g. 'age', 'ECOG performance status', 'ALT'.")
    comparator: Literal["<", "<=", ">", ">=", "=", "between", "other"]
    value: str = Field(description="The threshold as written, e.g. '18', '0-1', '2.5 x ULN'.")
    unit: Optional[str]


class LLMAtomicCriterion(BaseModel):
    type: CriterionType
    text: str = Field(description="Self-contained restatement of ONE checkable fact.")
    source_text: str = Field(
        description="The verbatim span of the original criteria text this item was derived from."
    )
    time_window: Optional[str] = Field(
        description="Time constraint as written, e.g. 'within 6 months before enrollment'. null if none."
    )
    threshold: Optional[Threshold]
    concepts: list[Concept]


class LLMDecomposition(BaseModel):
    criteria: list[LLMAtomicCriterion]


class LLMEvidence(BaseModel):
    segment_id: str = Field(description="ID of the note segment the quote comes from, e.g. 'S3'.")
    quote: str = Field(description="Exact, contiguous, character-for-character copy from that segment.")


class LLMJudgment(BaseModel):
    criterion_id: str
    verdict: Verdict
    evidence: list[LLMEvidence]
    patient_value: Optional[str] = Field(
        description="For numeric criteria: the patient's value and unit as written in the note. null otherwise."
    )
    rationale: str = Field(description="One short sentence.")
    confidence: Confidence
    missing_info: Optional[str] = Field(
        description="When verdict is 'unknown': what the clinician should look up or ask. null otherwise."
    )


class LLMJudgmentBatch(BaseModel):
    judgments: list[LLMJudgment]


# --------------------------------------------------------------------------
# Stored / served shapes
# --------------------------------------------------------------------------


class Segment(BaseModel):
    """A slice of the note. Invariant: ``note[start:end] == text``."""

    id: str
    start: int
    end: int
    text: str


class Patient(BaseModel):
    id: str
    source: str
    text: str
    note_date: Optional[str] = None  # ISO date the note was written, when known


class Trial(BaseModel):
    nct_id: str
    title: str
    conditions: list[str] = []
    criteria_text: str
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    sex: Optional[str] = None
    overall_status: Optional[str] = None
    url: str
    fetched_at: str
    # "ctgov_api" (current registry record) or "trec_snapshot_<date>" (what TREC assessors saw).
    source: str = "ctgov_api"


class AtomicCriterion(BaseModel):
    id: str  # INC-01, EXC-03 ...
    type: CriterionType
    text: str
    source_text: str
    # False when the model's source_text could not be found verbatim in the
    # trial's criteria text (shown as a warning; the criterion is still judged).
    source_verified: bool = True
    origin: Literal["criteria_text", "structured_field"] = "criteria_text"
    time_window: Optional[str] = None
    threshold: Optional[Threshold] = None
    concepts: list[Concept] = []


class Evidence(BaseModel):
    quote: str
    start: int
    end: int
    segment_id: str


class RetrievedSegment(BaseModel):
    segment_id: str
    score: float


class CriterionResult(BaseModel):
    criterion_id: str
    criterion_text: str
    source_text: str
    type: CriterionType
    verdict: Verdict
    effect: Effect
    evidence: list[Evidence]
    rationale: str
    confidence: Confidence
    patient_value: Optional[str] = None
    missing_info: Optional[str] = None
    time_window: Optional[str] = None
    threshold: Optional[Threshold] = None
    concepts: list[Concept] = []
    source_verified: bool = True
    origin: str = "criteria_text"
    retrieved_segments: list[RetrievedSegment] = []
    # Audit trail of what the program changed after the model answered.
    model_verdict: Verdict
    downgraded: bool = False
    rejected_quotes: list[str] = []
    flags: list[str] = []


class SummaryItem(BaseModel):
    criterion_id: str
    text: str


class Summary(BaseModel):
    overall: Overall
    overall_label: str
    counts: dict[str, int]
    key_reasons: list[SummaryItem]
    follow_up: list[SummaryItem]
    disclaimer: str


class CitationStats(BaseModel):
    quotes_total: int
    quotes_verified: int
    judgments_needing_evidence: int
    judgments_downgraded: int


class RunResult(BaseModel):
    run_id: str
    created_at: str
    model: str
    # "full": the whole note was given to the model; "retrieved": BM25-selected segments only.
    note_coverage: str = "full"
    judge_batch_size: int = 8
    patient: Patient
    segments: list[Segment]
    trial: Trial
    criteria: list[CriterionResult]
    summary: Summary
    citation_stats: CitationStats


class Review(BaseModel):
    run_id: str
    criterion_id: str
    action: Literal["agree", "modify"]
    model_verdict: Verdict
    reviewer_verdict: Verdict
    comment: str = ""
    reviewer: str = ""
    created_at: str
