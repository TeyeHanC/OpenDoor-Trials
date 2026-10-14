"""Step 3: judge each atomic criterion against the retrieved note segments."""

from __future__ import annotations

import json

from .llm import LLMRefusal, StructuredLLM
from .models import AtomicCriterion, LLMJudgment, LLMJudgmentBatch, Patient, Segment, Trial
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
"brain MRI showed no metastases"), and it covers everything it logically includes. "She has not \
received any systemic therapy for lung cancer" shows she has not received osimertinib, chemotherapy \
or any other systemic drug for it; "newly diagnosed" and "treatment-naive" show the disease is not a \
recurrence after earlier treatment. So when a criterion requires a history the note rules out in this \
way (prior adjuvant osimertinib, progression on a previous line), the verdict is "not_met" with that \
statement as the quote, not "unknown". Apply this the same way to inclusion and exclusion criteria: \
if a sentence is strong enough to clear an exclusion, it is strong enough to fail an inclusion that \
requires the opposite.
- The one exception is a conclusion that follows with certainty from a demographic fact the note \
states outright, such as age or sex: a man is not pregnant; a 45-year-old is not a child. Quote the \
stated fact and set `basis` to "inferred". Nothing else may be inferred. In particular, never \
conclude that a condition, treatment or event is absent because the note does not mention it, \
because a related test was normal, or because it would usually have been recorded (a normal \
creatinine does not establish the absence of kidney disease; no listed anticoagulant does not \
establish that none is taken). Those are "unknown".
- Applying medical definitions to what the note states is not inference, and you should do it \
freely: lung adenocarcinoma is a non-squamous non-small cell lung cancer; an ECOG performance status \
is the same scale as a WHO performance status; stage IV disease is metastatic. Such knowledge tells \
you what a stated fact means. What you may not do is supply a fact about this patient that the note \
does not give.
- Some criteria apply only to a subgroup: "female patients of childbearing potential must use \
contraception", "patients from Japan must be at least 20". When the note shows the patient is \
outside that subgroup, the criterion places no requirement on them and cannot keep them out: answer \
"met" for an inclusion criterion and "not_met" for an exclusion criterion, quote the fact that puts \
them outside the subgroup, set `basis` to "inferred" and say in the rationale that the criterion does \
not apply. Answering "not_met" to such an inclusion criterion would wrongly rule out every man from \
a trial that has a rule for women.
- Read each criterion as the trial's authors meant it, using the trial's title and the conditions \
it studies. A trial for patients with an EGFR mutation cannot be excluding patients for having that \
mutation: an exclusion such as "other driver alterations with an approved targeted therapy" refers \
to alterations other than the one the trial is about. If a reading of an exclusion would shut out \
the very population the trial enrolls, it is the wrong reading.
- A risk factor is not the disease, and a controlled condition is not an uncontrolled one. \
Well-controlled hypertension does not satisfy an exclusion for "cardiovascular disease" or for \
"uncontrolled hypertension"; diabetes on stable treatment is not "uncontrolled diabetes". When a \
criterion lists specific qualifying conditions, the patient must have one of those.
- A criterion about something that can only happen after enrollment or after study treatment \
starts (a test result three weeks into treatment, a response to the study drug) is not failed by a \
patient who has not started yet. Answer "unknown" and say it is assessed later.
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
`patient_value`, then compare it with the threshold. If the threshold depends on something the note \
lacks (such as the lab's upper limit of normal), say so and answer "unknown".

Arithmetic the program re-checks
When your verdict rests on date arithmetic or on comparing a number with a limit, also fill in the \
matching check with the inputs you used and your own answer. The program recomputes the answer from \
those inputs. If it agrees, the clinician sees a verified calculation; if it disagrees, your \
determination is set aside for manual review. So give the inputs exactly as the note and the \
criterion state them, and do not adjust them to fit your conclusion.
- `time_check`, for a look-back window ("within 28 days", "in the last 6 months"): the event's date \
and the reference date as YYYY-MM-DD, the window, and whether you find the event inside it. Use it \
only when the note gives the event's full date; with only a month or a year, leave it null and judge \
the timing as far as the note allows.
- `threshold_check`, for a number against a limit: the patient's value and unit from the note, the \
comparator and limit from the criterion, and whether the comparison holds. Choose the comparator so \
that "value comparator threshold" restates the criterion (for "ANC below 1.5", comparator "<"). When \
the limit is the upper or lower limit of normal, or a multiple of it, put the multiplier in \
`threshold` and the lab's limit in `reference_limit`: "ALT above 2.5 x ULN" with a ULN of 40 is \
threshold 2.5 and reference_limit 40, and a plain "platelets below LLN" with an LLN of 150 is \
threshold 1 and reference_limit 150 (the limit goes in `reference_limit` only, never in both). Use the limit the note states when it gives one. When it does \
not, use the typical adult limit from the table below and set `reference_limit_assumed` to true: \
reference ranges differ between laboratories, so the program accepts the conclusion only if it would \
still hold with a limit 25% lower or higher, shows the clinician that the limit was assumed, and \
otherwise sets the determination aside. For a value with no entry in the table, answer "unknown". \
When the units differ, keep each as written and name the `analyte`; the program converts.
  Typical adult reference limits (assumptions, to be flagged):
  ALT upper limit 40 U/L; AST upper limit 40 U/L; alkaline phosphatase upper limit 120 U/L;
  total bilirubin upper limit 1.2 mg/dL (20.5 umol/L); absolute neutrophil count lower limit 1.5 x 10^9/L;
  platelets lower limit 150 x 10^9/L; white blood cell count lower limit 4.0 x 10^9/L.
Leave both null for criteria that involve no arithmetic. A check supports a verdict but does not \
replace the quote: every value and date in it must come from text you can quote.

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
- `established_part`: for "unknown" only. When the note settles part of a criterion and leaves the \
rest open, name the settled part in a few words ("no brain metastases on MRI", "denies ILD history"), \
so the clinician sees what is already covered and `missing_info` can name only what is left. Null \
when the note settles nothing.

Return exactly one judgment per criterion, using the criterion ids you were given."""

# Appended to the system prompt only when the note-search tools are offered.
NOTE_TOOL_GUIDANCE = """
This note is too long to show in full, so you were given the excerpts a keyword search judged most \
relevant. That search matches words rather than meaning and can miss a passage that uses a synonym or \
an abbreviation. Before answering "unknown" for a criterion, use search_note with alternative wordings, \
and read_segments to see the lines around a match. Only after that is something absent from the note \
as far as you can tell, and the verdict is then still "unknown", never "not_met"."""


def system_prompt(tools: list[Tool]) -> str:
    if any(tool.name == "search_note" for tool in tools):
        return SYSTEM_PROMPT + "\n" + NOTE_TOOL_GUIDANCE
    return SYSTEM_PROMPT


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


def render_criteria(criteria: list[AtomicCriterion], trial: Trial | None = None) -> str:
    context = ""
    if trial is not None:
        # What the trial is for, so a criterion is read as its authors meant it.
        context = (
            f"Trial: {trial.nct_id} - {trial.title}\n"
            f"Conditions studied: {', '.join(trial.conditions) or 'not listed'}\n\n"
        )
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
    return context + "Criteria to judge:\n" + json.dumps(items, indent=2, ensure_ascii=False)


def judge_batch(
    patient: Patient,
    segments: list[Segment],
    criteria: list[AtomicCriterion],
    llm: StructuredLLM,
    *,
    complete: bool,
    tools: list[Tool] | None = None,
    trace: list[dict] | None = None,
    trial: Trial | None = None,
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
                render_criteria(criteria, trial),
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
