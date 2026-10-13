"""Step 3: judge each atomic criterion against the retrieved note segments."""

from __future__ import annotations

import json

from .llm import LLMRefusal, StructuredLLM
from .models import AtomicCriterion, LLMJudgment, LLMJudgmentBatch, Patient, Segment
from .tools import Tool

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
- The one exception is a conclusion that follows with certainty from a demographic fact the note \
states outright, such as age or sex: a man is not pregnant; a 45-year-old is not a child. Quote the \
stated fact and set `basis` to "inferred". Nothing else may be inferred. In particular, never \
conclude that a condition, treatment or event is absent because the note does not mention it, \
because a related test was normal, or because it would usually have been recorded (a normal \
creatinine does not establish the absence of kidney disease; no listed anticoagulant does not \
establish that none is taken). Those are "unknown".
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
- `basis`: "stated" when your quote itself is about what the criterion asks (including a stated \
value compared with a threshold); "inferred" when your quote is a different fact, age or sex, from \
which the verdict follows; "not_applicable" for "unknown". Inferred determinations are listed \
separately for the clinician to double-check.
- `rationale`: one short sentence explaining the verdict in plain language.
- `confidence`: "high" when the note states the fact directly; "medium" when a small, safe \
interpretation was needed; "low" when the note is ambiguous or conflicting, or the answer depends on \
an assumption about timing, units or meaning. Low-confidence items are shown to the clinician first.
- `missing_info`: for "unknown" only, what the clinician should look up or ask the patient; else null.

Return exactly one judgment per criterion, using the criterion ids you were given."""

# Appended to the system prompt only when tools are offered.
CALCULATOR_GUIDANCE = """

Tools
You have calculator tools for dates, thresholds and lab units. Use them whenever a verdict depends on \
arithmetic: whether an event falls inside a time window, how a value compares with a limit, or a \
comparison across different units. The clinician can re-check a computed result, which they cannot do \
with mental arithmetic. A tool result is a calculation, not evidence: the dates and values you pass in \
must come from the note, and your quotes must still be the note's own text. Criteria that need no \
arithmetic need no tool call."""

NOTE_TOOL_GUIDANCE = """
This note is too long to show in full, so you were given the excerpts a keyword search judged most \
relevant. That search matches words rather than meaning and can miss a passage that uses a synonym or \
an abbreviation. Before answering "unknown" for a criterion, use search_note with alternative wordings, \
and read_segments to see the lines around a match. Only after that is something absent from the note \
as far as you can tell, and the verdict is then still "unknown", never "not_met"."""


def system_prompt(tools: list[Tool]) -> str:
    names = {tool.name for tool in tools}
    prompt = SYSTEM_PROMPT
    if names:
        prompt += CALCULATOR_GUIDANCE
    if "search_note" in names:
        prompt += NOTE_TOOL_GUIDANCE
    return prompt


def render_note(patient: Patient, segments: list[Segment], *, complete: bool, searchable: bool = False) -> str:
    if complete:
        coverage = "Coverage: the complete note"
    elif searchable:
        coverage = "Coverage: excerpts only; use search_note and read_segments to see the rest of the note"
    else:
        coverage = "Coverage: excerpts only; the rest of the note is not shown"
    header = [
        "The note below is a synthetic or publicly released patient description.",
        f"Note date: {patient.note_date or 'not given'}",
        coverage,
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
    tools: list[Tool] | None = None,
    trace: list[dict] | None = None,
) -> dict[str, LLMJudgment] | None:
    """Judgments keyed by criterion id; None when the model declined the batch.

    Tool calls the model makes on the way are appended to ``trace``.
    """
    tools = tools or []
    searchable = any(tool.name == "search_note" for tool in tools)
    try:
        result = llm.structured(
            system=system_prompt(tools),
            user_blocks=[
                render_note(patient, segments, complete=complete, searchable=searchable),
                render_criteria(criteria),
            ],
            schema=LLMJudgmentBatch,
            tools=tools,
            trace=trace,
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
