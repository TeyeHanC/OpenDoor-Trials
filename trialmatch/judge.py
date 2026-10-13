"""Step 3: judge each atomic criterion against the retrieved note segments."""

from __future__ import annotations

import json

from .llm import LLMRefusal, StructuredLLM
from .models import AtomicCriterion, LLMJudgment, LLMJudgmentBatch, Patient, Segment

SYSTEM_PROMPT = """\
You are helping screen a patient for a clinical trial. You will be given segments of the patient's \
medical note and a list of eligibility criteria, and you decide, for each criterion, whether the \
note shows the patient meets it.

Your output goes to an outpatient physician or research coordinator who will check every \
determination against the note before acting on it, and who makes the enrollment decision. They \
will not trust a conclusion they cannot trace to the note, and a wrong "not eligible" costs a \
patient a trial they could have joined. So the two things that matter most are: never state \
something the note does not say, and always show exactly where in the note your answer comes from.

Verdicts
Judge each criterion exactly as it is written, whether it is an inclusion or an exclusion criterion.
- "met": the note shows the criterion's statement is true of the patient.
- "not_met": the note shows the criterion's statement is false of the patient.
- "unknown": the note does not contain enough information to tell.
For an exclusion criterion, "met" therefore means the patient HAS the excluding condition (which \
will block enrollment) and "not_met" means the note shows they do not have it. Do not invert this \
yourself; the program handles what each verdict means for enrollment.

Use only what the note states
- If the note does not mention something, the verdict is "unknown". Silence is not evidence of \
absence: a note that never mentions hepatitis does not show the patient is free of hepatitis, and a \
medication list that omits a drug does not show the patient never took it.
- An explicit negative statement IS evidence ("denies chest pain", "no history of seizures", \
"brain MRI showed no metastases").
- You may rely on direct logical consequences of what is written (a 45-year-old meets "age 18 or \
older"; a man is not pregnant). Do not make clinical guesses beyond that (a normal creatinine does \
not establish the absence of all kidney disease).
- Criteria about consent, willingness, life expectancy or investigator judgment are normally \
"unknown" unless the note addresses them directly.
- When only excerpts of the note are provided, information outside them is unavailable to you; \
treat it as unknown.

Negation, family history and time
- Negated findings ("denies diabetes") are not present conditions.
- Family history ("mother had breast cancer") says nothing about the patient's own history.
- Past events are not current states. "Chemotherapy five years ago" does not mean the patient is on \
chemotherapy now, though it does count as prior chemotherapy.
- For criteria with a time window, work out the dates. Use the note date when one is given; \
otherwise treat the note as describing the present and use its relative expressions ("two weeks \
ago"). If the timing cannot be established well enough to place the event inside or outside the \
window, the verdict is "unknown".

Numeric criteria
For lab values, age, doses and scores, put the patient's value and unit as written in the note in \
`patient_value`, then compare it with the threshold. If the units differ and cannot be converted \
with certainty, or the threshold depends on something the note lacks (such as the lab's upper limit \
of normal), say so and answer "unknown" or lower your confidence.

Evidence
- Every "met" or "not_met" needs at least one quote that supports it.
- A quote must be an exact, contiguous copy of text from one segment: same characters, case and \
punctuation, no ellipses, no paraphrase, nothing added. A program checks each quote against the \
note character by character, and a determination whose quotes cannot be found is discarded.
- Quote the shortest span that carries the fact, usually a phrase or one sentence.
- For "unknown", leave evidence empty unless a passage is relevant but inconclusive, in which case \
quote it so the clinician can see why it falls short.

Other fields
- `rationale`: one short sentence explaining the verdict in plain language.
- `confidence`: "high" when the note states the fact directly; "medium" when a small, safe \
interpretation was needed; "low" when the note is ambiguous or conflicting, or the answer depends on \
an assumption about timing, units or meaning. Low-confidence items are shown to the clinician first.
- `missing_info`: for "unknown" only, what the clinician should look up or ask the patient; else null.

Return exactly one judgment per criterion, using the criterion ids you were given."""


def render_note(patient: Patient, segments: list[Segment], *, complete: bool) -> str:
    header = [
        "The note below is a synthetic or publicly released patient description.",
        f"Note date: {patient.note_date or 'not given'}",
        "Coverage: the complete note" if complete else "Coverage: excerpts only; the rest of the note is not shown",
    ]
    body = "\n".join(f"[{s.id}] {s.text}" for s in segments)
    return "\n".join(header) + f"\n\n<note>\n{body}\n</note>"


def render_criteria(criteria: list[AtomicCriterion]) -> str:
    items = []
    for c in criteria:
        item = {"criterion_id": c.id, "type": c.type, "criterion": c.text}
        if c.source_text and c.source_text != c.text:
            item["original_wording"] = c.source_text
        if c.time_window:
            item["time_window"] = c.time_window
        if c.threshold:
            item["threshold"] = c.threshold.model_dump(exclude_none=True)
        items.append(item)
    return "Criteria to judge:\n" + json.dumps(items, indent=2, ensure_ascii=False)


def judge_batch(
    patient: Patient,
    segments: list[Segment],
    criteria: list[AtomicCriterion],
    llm: StructuredLLM,
    *,
    complete: bool,
) -> dict[str, LLMJudgment] | None:
    """Judgments keyed by criterion id; None when the model declined the batch."""
    try:
        result = llm.structured(
            system=SYSTEM_PROMPT,
            user_blocks=[render_note(patient, segments, complete=complete), render_criteria(criteria)],
            schema=LLMJudgmentBatch,
        )
    except LLMRefusal:
        return None
    wanted = {c.id for c in criteria}
    judgments: dict[str, LLMJudgment] = {}
    for judgment in result.judgments:
        criterion_id = judgment.criterion_id.strip()
        if criterion_id in wanted and criterion_id not in judgments:
            judgments[criterion_id] = judgment
    return judgments
