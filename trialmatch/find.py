"""Trial finder: search ClinicalTrials.gov for trials a patient might qualify for.

This is the one step where the model drives the work: it decides what to search
for, reads the results, refines the query and reads the criteria of promising
trials before it answers. The output is a shortlist, not a determination; each
shortlisted trial still goes through the criterion-by-criterion match
(``pipeline.run_match``) before a clinician sees a verdict.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .config import Settings
from .ctgov import NCT_RE
from .llm import StructuredLLM
from .models import Patient
from .tools import trial_tools

SYSTEM_PROMPT = """\
You help a physician find clinical trials that a patient might be able to join. You are given the \
patient's medical note and tools to search the ClinicalTrials.gov registry. Your shortlist goes to a \
second step that checks each trial's eligibility criteria one by one against the note, and then to the \
physician, who decides. So your job is to find the right trials to check, not to rule on eligibility.

How to search
- Start from the patient's main diagnosis, then narrow with what makes this patient specific: stage, \
biomarkers, prior treatment, line of therapy. Run several searches with different wordings; a registry \
entry may say "NSCLC" where the note says "lung adenocarcinoma".
- Prefer trials that are recruiting. Use the country filter only if you were told where the patient is.
- Before you shortlist a trial, read its criteria with get_trial_criteria. Titles are often misleading \
about who can enroll.
- Search queries leave this system. Send clinical terms only, never a name, a date, a record number or \
any other detail that could identify a person.

What belongs on the shortlist
- Trials for this patient's condition and treatment setting whose criteria the note does not plainly \
contradict. If the note clearly rules a trial out (wrong mutation, a treatment the patient has not had \
that the trial requires, an age limit), leave it off.
- When the note is silent on a criterion, that is not a reason to drop the trial; note it under \
`concerns` so the physician knows what to check.
- A patient who could join is easier to lose than an extra trial is to review: when in doubt, include \
the trial and say why you are unsure.
- List only trials your tools returned in this conversation, by their NCT number. The program discards \
any other id. A shorter list, or an empty one, is the right answer when the registry has little to offer.

Fields
- `candidates`: best match first, at most the number you were asked for.
- `why_relevant`: one or two sentences tying the trial to facts in the note.
- `concerns`: what might still exclude the patient or needs checking; null if nothing stands out.
- `search_notes`: a short account of what you searched for and anything the physician should know \
about the search, such as a query that returned nothing."""


class LLMCandidate(BaseModel):
    nct_id: str
    why_relevant: str
    concerns: Optional[str]


class LLMShortlist(BaseModel):
    candidates: list[LLMCandidate]
    search_notes: str = Field(description="What was searched and anything notable about the search.")


class Candidate(BaseModel):
    nct_id: str
    title: str
    status: Optional[str] = None
    why_relevant: str
    concerns: Optional[str] = None


class FindResult(BaseModel):
    patient_id: str
    created_at: str
    model: str
    candidates: list[Candidate]
    # Trial ids the model named that no tool call had returned; dropped, kept here for audit.
    rejected_ids: list[str] = []
    search_notes: str
    tool_calls: list[dict] = []


def find_trials(
    patient: Patient, settings: Settings, llm: StructuredLLM, *, max_candidates: int = 5, country: str | None = None
) -> FindResult:
    seen: dict[str, dict] = {}
    trace: list[dict] = []
    request = f"Find up to {max_candidates} trials to check for this patient."
    if country:
        request += f" The patient is in {country}."
    shortlist = llm.structured(
        system=SYSTEM_PROMPT,
        user_blocks=[
            "The note below is a synthetic or publicly released patient description.\n"
            f"Note date: {patient.note_date or 'not given'}\n\n<note>\n{patient.text}\n</note>",
            request,
        ],
        schema=LLMShortlist,
        tools=trial_tools(settings, seen),
        trace=trace,
    )

    candidates, rejected, listed = [], [], set()
    for item in shortlist.candidates:
        nct_id = item.nct_id.strip().upper()
        if not NCT_RE.match(nct_id) or nct_id not in seen:
            rejected.append(item.nct_id)
            continue
        if nct_id in listed:
            continue
        listed.add(nct_id)
        candidates.append(
            Candidate(
                nct_id=nct_id,
                title=seen[nct_id].get("title") or "",
                status=seen[nct_id].get("status"),
                why_relevant=item.why_relevant.strip(),
                concerns=(item.concerns or "").strip() or None,
            )
        )

    return FindResult(
        patient_id=patient.id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=settings.model,
        candidates=candidates[:max_candidates],
        rejected_ids=rejected,
        search_notes=shortlist.search_notes.strip(),
        tool_calls=trace,
    )


def save_find_result(settings: Settings, result: FindResult):
    path = settings.data_dir / "find" / f"{result.patient_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def format_find_result(result: FindResult) -> str:
    searches = sum(1 for call in result.tool_calls if call["tool"] == "search_trials")
    read = sum(1 for call in result.tool_calls if call["tool"] == "get_trial_criteria")
    lines = [
        f"{result.patient_id}: {len(result.candidates)} candidate trial(s) "
        f"({searches} registry searches, {read} criteria read)",
    ]
    for i, c in enumerate(result.candidates, start=1):
        lines += [f"\n{i}. {c.nct_id}  {c.title}  [{c.status or 'status unknown'}]", f"   Why: {c.why_relevant}"]
        if c.concerns:
            lines.append(f"   To check: {c.concerns}")
    if result.rejected_ids:
        lines.append(f"\nDropped (named by the model but never returned by a search): {', '.join(result.rejected_ids)}")
    lines += [f"\nSearch notes: {result.search_notes}", "\nA shortlist only: run the criterion-level match on each trial before relying on it."]
    return "\n".join(lines)
