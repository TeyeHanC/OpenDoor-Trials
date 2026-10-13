"""Step 4: roll criterion-level results up into an overall screening conclusion.

The roll-up is rule-based on purpose: the overall conclusion can then be
re-derived by hand from the criterion list, and nothing in the summary can say
more than the verified criterion results do.
"""

from __future__ import annotations

from .models import OVERALL_LABELS, CriterionResult, Overall, Summary, SummaryItem

DISCLAIMER = (
    "Screening aid only. This tool does not make enrollment decisions; every determination must be "
    "reviewed and confirmed by a physician."
)
MAX_KEY_REASONS = 5
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _blocker_text(c: CriterionResult) -> str:
    lead = "Exclusion criterion applies" if c.type == "exclusion" else "Inclusion criterion not met"
    return f"{lead}: {c.criterion_text} ({c.rationale})"


def _support_text(c: CriterionResult) -> str:
    lead = "Meets inclusion criterion" if c.type == "inclusion" else "Exclusion criterion does not apply"
    return f"{lead}: {c.criterion_text} ({c.rationale})"


def _follow_up_text(c: CriterionResult) -> str:
    kind = "inclusion" if c.type == "inclusion" else "exclusion"
    if c.verdict == "unknown":
        detail = c.missing_info or "Not documented in the note."
        return f"Check {kind} criterion \"{c.criterion_text}\": {detail}"
    return f"Verify low-confidence {kind} determination \"{c.criterion_text}\": {c.rationale}"


def summarize(criteria: list[CriterionResult]) -> Summary:
    blockers = [c for c in criteria if c.effect == "blocks"]
    unresolved = [c for c in criteria if c.effect == "unresolved"]
    supports = [c for c in criteria if c.effect == "supports"]

    overall: Overall
    if blockers:
        overall = "not_eligible"
    elif unresolved:
        overall = "needs_more_info"
    else:
        overall = "possibly_eligible"

    by_confidence = lambda c: _CONFIDENCE_ORDER[c.confidence]  # noqa: E731
    if blockers:
        key = [(c, _blocker_text(c)) for c in sorted(blockers, key=by_confidence)]
    else:
        inclusion_first = sorted(supports, key=lambda c: (c.type != "inclusion", by_confidence(c)))
        key = [(c, _support_text(c)) for c in inclusion_first]

    # Low-confidence determinations first: they are the ones most likely to be wrong.
    low_confidence = [c for c in criteria if c.verdict != "unknown" and c.confidence == "low"]
    follow_up = low_confidence + unresolved

    return Summary(
        overall=overall,
        overall_label=OVERALL_LABELS[overall],
        counts={
            "total": len(criteria),
            "supports": len(supports),
            "blocks": len(blockers),
            "unresolved": len(unresolved),
            "low_confidence": len([c for c in criteria if c.confidence == "low"]),
            "downgraded": len([c for c in criteria if c.downgraded]),
        },
        key_reasons=[SummaryItem(criterion_id=c.criterion_id, text=text) for c, text in key[:MAX_KEY_REASONS]],
        follow_up=[SummaryItem(criterion_id=c.criterion_id, text=_follow_up_text(c)) for c in follow_up],
        disclaimer=DISCLAIMER,
    )
