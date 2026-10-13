"""Citation check: every quote must exist verbatim in the note.

The model is not trusted to report offsets or to copy text faithfully. Each
quote is searched for in the original note; the stored evidence is the note's
own text at the matched span. A "met" / "not met" judgment left without any
verified quote is downgraded to "unknown" and flagged.
"""

from __future__ import annotations

import re

from .checks import run_checks
from .models import AtomicCriterion, CriterionResult, Evidence, LLMJudgment, RetrievedSegment, Segment, effect_of


def locate_quote(text: str, quote: str, within: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Find ``quote`` in ``text`` and return its (start, end), or None.

    Matching is character-exact except that any run of whitespace in the quote
    may match any run of whitespace in the text (a line break versus a space is
    not a change of content). The span given by ``within`` is searched first so
    a phrase that occurs twice resolves to the segment the model pointed at.
    """
    tokens = quote.split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    if within is not None:
        match = pattern.search(text, within[0], within[1])
        if match:
            return match.start(), match.end()
    match = pattern.search(text)
    return (match.start(), match.end()) if match else None


def _segment_containing(segments: list[Segment], start: int) -> str:
    for segment in segments:
        if segment.start <= start < segment.end:
            return segment.id
    return ""


def verify_judgment(
    judgment: LLMJudgment,
    criterion: AtomicCriterion,
    note: str,
    segments: list[Segment],
    retrieved: list[RetrievedSegment],
) -> CriterionResult:
    by_id = {s.id: s for s in segments}
    evidence: list[Evidence] = []
    rejected: list[str] = []
    seen: set[tuple[int, int]] = set()
    for item in judgment.evidence:
        cited = by_id.get(item.segment_id.strip())
        span = locate_quote(note, item.quote, (cited.start, cited.end) if cited else None)
        if span is None:
            rejected.append(item.quote)
        elif span not in seen:
            seen.add(span)
            evidence.append(
                Evidence(quote=note[span[0] : span[1]], start=span[0], end=span[1], segment_id=_segment_containing(segments, span[0]))
            )

    verdict, confidence, flags = judgment.verdict, judgment.confidence, []
    missing_info = judgment.missing_info
    if rejected:
        flags.append("quote_not_found_in_note")
    downgraded = verdict != "unknown" and not evidence
    if downgraded:
        flags.append("downgraded_no_verified_evidence")
        missing_info = (
            f"The model answered '{verdict}' but its supporting quote could not be found in the note, "
            "so this needs a manual check."
        )
        verdict, confidence = "unknown", "low"

    checks = run_checks(judgment.time_check, judgment.threshold_check, note)
    wrong = [c for c in checks if c.status == "mismatch"]
    if wrong and verdict != "unknown":
        # The model's arithmetic does not survive recomputation; do not pass its verdict on.
        flags.append("calculation_mismatch")
        missing_info = (
            f"The model answered '{verdict}' but its arithmetic did not match the program's ({wrong[0].detail}), "
            "so this needs a manual check."
        )
        verdict, confidence = "unknown", "low"
    elif verdict != "unknown":
        if any(c.status == "not_checkable" for c in checks):
            flags.append("calculation_not_checkable")
            confidence = "low"
        if any(c.inputs_not_in_note for c in checks):
            flags.append("calculation_input_not_in_note")
            confidence = "low"

    return CriterionResult(
        criterion_id=criterion.id,
        criterion_text=criterion.text,
        source_text=criterion.source_text,
        type=criterion.type,
        verdict=verdict,
        effect=effect_of(criterion.type, verdict),
        evidence=sorted(evidence, key=lambda e: e.start),
        rationale=judgment.rationale.strip(),
        confidence=confidence,
        basis="not_applicable" if verdict == "unknown" else judgment.basis,
        patient_value=judgment.patient_value,
        missing_info=missing_info if verdict == "unknown" else None,
        time_window=criterion.time_window,
        threshold=criterion.threshold,
        concepts=criterion.concepts,
        source_verified=criterion.source_verified,
        origin=criterion.origin,
        retrieved_segments=retrieved,
        checks=[c.model_dump() for c in checks],
        model_verdict=judgment.verdict,
        downgraded=downgraded,
        rejected_quotes=rejected,
        flags=flags,
    )


def unresolved_result(
    criterion: AtomicCriterion, retrieved: list[RetrievedSegment], *, flag: str, reason: str
) -> CriterionResult:
    """Placeholder for a criterion the model produced no usable judgment for."""
    return CriterionResult(
        criterion_id=criterion.id,
        criterion_text=criterion.text,
        source_text=criterion.source_text,
        type=criterion.type,
        verdict="unknown",
        effect="unresolved",
        evidence=[],
        rationale=reason,
        confidence="low",
        basis="not_applicable",
        missing_info="No automated assessment is available for this criterion; check it manually.",
        time_window=criterion.time_window,
        threshold=criterion.threshold,
        concepts=criterion.concepts,
        source_verified=criterion.source_verified,
        origin=criterion.origin,
        retrieved_segments=retrieved,
        model_verdict="unknown",
        flags=[flag],
    )
