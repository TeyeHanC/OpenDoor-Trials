"""Tools the model may call while it works.

Three groups, each used where it earns its place:

* calculators   date and threshold arithmetic, so a time window or "2.5 x ULN"
                is computed by the program and can be re-checked by a clinician
* note tools    search and read a note that was too long to send whole
* trial tools   search ClinicalTrials.gov and read a trial's criteria (used by
                the trial finder, see ``find.py``)

Every tool validates its input with a pydantic model and never raises to the
caller: a bad input or a failed look-up comes back as an error result the model
can read and recover from.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import date
from typing import Callable, Literal, Optional

import httpx
from anthropic import transform_schema
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .ctgov import TrialNotFound, fetch_trial
from .models import Segment
from .retrieval import BM25Index


class ToolError(Exception):
    """A failure the model should be told about (bad input, nothing found, service down)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], dict]

    def definition(self) -> dict:
        schema = transform_schema(self.input_model)
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema, "strict": True}

    def run(self, raw_input: object) -> tuple[str, bool]:
        """Return (result text, is_error)."""
        try:
            args = self.input_model.model_validate(raw_input)
            return json.dumps(self.handler(args), ensure_ascii=False), False
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return f"Invalid input: {problems}", True
        except ToolError as exc:
            return str(exc), True
        except httpx.HTTPError as exc:
            return f"The request failed ({type(exc).__name__}). Try again or continue without it.", True


# --------------------------------------------------------------------------
# Calculators
# --------------------------------------------------------------------------


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolError(
            f"{field} must be a full date written YYYY-MM-DD, got {value!r}. If the note gives only a month "
            "or a year, the exact timing is not established."
        ) from None


def _shift_back(day: date, amount: int, unit: str) -> date:
    if unit == "days":
        return date.fromordinal(day.toordinal() - amount)
    if unit == "weeks":
        return date.fromordinal(day.toordinal() - 7 * amount)
    months = amount if unit == "months" else 12 * amount
    index = day.year * 12 + (day.month - 1) - months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


class DateDifferenceInput(BaseModel):
    earlier_date: str = Field(description="YYYY-MM-DD")
    later_date: str = Field(description="YYYY-MM-DD")


def date_difference(args: DateDifferenceInput) -> dict:
    earlier, later = _parse_date(args.earlier_date, "earlier_date"), _parse_date(args.later_date, "later_date")
    days = (later - earlier).days
    return {"days": days, "weeks": round(days / 7, 1), "months": round(days / 30.44, 1), "years": round(days / 365.25, 2)}


class TimeWindowInput(BaseModel):
    event_date: str = Field(description="Date of the event in the note, YYYY-MM-DD.")
    reference_date: str = Field(description="Date the window is counted back from (normally the note date), YYYY-MM-DD.")
    amount: int = Field(description="Length of the window, e.g. 28 for 'within 28 days'.")
    unit: Literal["days", "weeks", "months", "years"]


def check_time_window(args: TimeWindowInput) -> dict:
    if args.amount < 0:
        raise ToolError("amount must not be negative")
    event, reference = _parse_date(args.event_date, "event_date"), _parse_date(args.reference_date, "reference_date")
    window_start = _shift_back(reference, args.amount, args.unit)
    return {
        "window_start": window_start.isoformat(),
        "window_end": reference.isoformat(),
        "days_before_reference": (reference - event).days,
        "event_is_within_window": window_start <= event <= reference,
        "event_is_after_reference": event > reference,
    }


class ThresholdInput(BaseModel):
    patient_value: float
    comparator: Literal["<", "<=", ">", ">=", "=", "between"]
    threshold: float = Field(description="The limit; the lower bound for 'between'. A multiplier when upper_limit_of_normal is given.")
    threshold_upper: Optional[float] = Field(description="Upper bound for 'between' (inclusive); null otherwise.")
    upper_limit_of_normal: Optional[float] = Field(
        description="Set only when the criterion is a multiple of ULN ('<= 2.5 x ULN') AND the note states the "
        "lab's ULN in the same unit as patient_value. null otherwise."
    )


def compare_to_threshold(args: ThresholdInput) -> dict:
    scale = args.upper_limit_of_normal if args.upper_limit_of_normal is not None else 1.0
    lower = args.threshold * scale
    value = args.patient_value
    if args.comparator == "between":
        if args.threshold_upper is None:
            raise ToolError("threshold_upper is required for 'between'")
        upper = args.threshold_upper * scale
        return {"effective_range": [lower, upper], "satisfied": lower <= value <= upper}
    satisfied = {
        "<": value < lower, "<=": value <= lower, ">": value > lower, ">=": value >= lower, "=": value == lower,
    }[args.comparator]
    return {"effective_threshold": lower, "satisfied": satisfied}


# Factor that converts one unit of the key into the analyte's reference unit (the first key).
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "creatinine": {"mg/dl": 1.0, "umol/l": 1 / 88.42},
    "bilirubin": {"mg/dl": 1.0, "umol/l": 1 / 17.104},
    "hemoglobin": {"g/dl": 1.0, "g/l": 0.1, "mmol/l": 1 / 0.6206},
    "albumin": {"g/dl": 1.0, "g/l": 0.1},
    "glucose": {"mg/dl": 1.0, "mmol/l": 18.016},
    "calcium": {"mg/dl": 1.0, "mmol/l": 1 / 0.2495},
    "blood_cell_count": {"10^9/l": 1.0, "10^3/ul": 1.0, "k/ul": 1.0, "/ul": 0.001, "/mm3": 0.001},
}


def _normalize_unit(unit: str) -> str:
    unit = unit.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    unit = unit.replace("x10", "10").replace("×10", "10").replace("cells", "")
    return unit.replace("mm^3", "mm3").replace("10e9", "10^9").replace("10e3", "10^3").replace("10*9", "10^9")


class ConvertUnitInput(BaseModel):
    analyte: Literal["creatinine", "bilirubin", "hemoglobin", "albumin", "glucose", "calcium", "blood_cell_count"] = Field(
        description="blood_cell_count covers neutrophils, platelets, white cells and lymphocytes."
    )
    value: float
    from_unit: str = Field(description="Unit as written in the note, e.g. 'umol/L'.")
    to_unit: str = Field(description="Unit used by the criterion, e.g. 'mg/dL'.")


def convert_lab_unit(args: ConvertUnitInput) -> dict:
    factors = _UNIT_FACTORS[args.analyte]
    source, target = _normalize_unit(args.from_unit), _normalize_unit(args.to_unit)
    for unit in (source, target):
        if unit not in factors:
            raise ToolError(
                f"No conversion known for {args.analyte} in {unit!r} (known: {', '.join(factors)}). "
                "Do not convert by hand; treat the comparison as not established."
            )
    converted = args.value * factors[source] / factors[target]
    return {"value": float(f"{converted:.4g}"), "unit": args.to_unit}


def calculator_tools() -> list[Tool]:
    return [
        Tool(
            "date_difference",
            "Exact time between two dates. Use it whenever a criterion depends on how long ago something "
            "happened, instead of working the dates out yourself.",
            DateDifferenceInput,
            date_difference,
        ),
        Tool(
            "check_time_window",
            "Whether an event falls inside a look-back window such as 'within 28 days' or 'in the last 6 "
            "months', counted back from a reference date. Months and years are calendar units.",
            TimeWindowInput,
            check_time_window,
        ),
        Tool(
            "compare_to_threshold",
            "Compare a patient's numeric value with a criterion's limit, including limits written as a "
            "multiple of the upper limit of normal. Both numbers must be in the same unit; convert first if not.",
            ThresholdInput,
            compare_to_threshold,
        ),
        Tool(
            "convert_lab_unit",
            "Convert a lab value between common units when the note and the criterion use different ones.",
            ConvertUnitInput,
            convert_lab_unit,
        ),
    ]


# --------------------------------------------------------------------------
# Note tools (long notes only)
# --------------------------------------------------------------------------


class SearchNoteInput(BaseModel):
    query: str = Field(
        description="Words to look for. Include synonyms, abbreviations and drug names in one query, "
        "e.g. 'interstitial lung disease ILD pneumonitis fibrosis'."
    )


class ReadSegmentsInput(BaseModel):
    first_segment: int = Field(description="Number of the first segment to read, e.g. 12 for S12.")
    last_segment: int = Field(description="Number of the last segment to read (inclusive).")


def note_tools(segments: list[Segment], max_results: int = 8, max_read: int = 20) -> list[Tool]:
    index = BM25Index(segments)
    by_id = {s.id: s for s in segments}

    def search_note(args: SearchNoteInput) -> dict:
        hits = [r for r in index.rank(args.query) if r.score > 0][:max_results]
        return {
            "segments_in_note": len(segments),
            "matches": [{"segment_id": r.segment_id, "text": by_id[r.segment_id].text} for r in hits],
        }

    def read_segments(args: ReadSegmentsInput) -> dict:
        first, last = max(1, args.first_segment), min(len(segments), args.last_segment)
        if first > last:
            raise ToolError(f"No such segments; the note has S1 to S{len(segments)}.")
        if last - first + 1 > max_read:
            raise ToolError(f"Read at most {max_read} segments per call.")
        return {"segments": [{"segment_id": s.id, "text": s.text} for s in segments[first - 1 : last]]}

    return [
        Tool(
            "search_note",
            "Search the full patient note by keyword and get the matching segments. The search matches "
            "words, not meaning, so try synonyms and abbreviations before concluding something is not there.",
            SearchNoteInput,
            search_note,
        ),
        Tool(
            "read_segments",
            "Read consecutive segments of the patient note by number, for example the lines around a "
            "search match or a whole section.",
            ReadSegmentsInput,
            read_segments,
        ),
    ]


# --------------------------------------------------------------------------
# Trial tools (trial finder)
# --------------------------------------------------------------------------

SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"
SEARCH_FIELDS = "NCTId,BriefTitle,OverallStatus,Condition,Phase,MinimumAge,MaximumAge,Sex,InterventionName"
MAX_PAGE_SIZE = 20


def summarize_study(payload: dict) -> dict:
    protocol = payload.get("protocolSection", {})
    elig = protocol.get("eligibilityModule", {})
    return {
        "nct_id": protocol.get("identificationModule", {}).get("nctId", ""),
        "title": protocol.get("identificationModule", {}).get("briefTitle", ""),
        "status": protocol.get("statusModule", {}).get("overallStatus"),
        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        "phases": protocol.get("designModule", {}).get("phases", []),
        "interventions": [i.get("name") for i in protocol.get("armsInterventionsModule", {}).get("interventions", [])],
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
    }


class SearchTrialsInput(BaseModel):
    condition: str = Field(description="Disease or condition, e.g. 'non-small cell lung cancer'.")
    other_terms: Optional[str] = Field(
        description="Further clinical terms: biomarker, drug, line of therapy, e.g. 'EGFR exon 19 first-line'. null if none."
    )
    country: Optional[str] = Field(description="Country to restrict trial sites to, e.g. 'United States'. null for anywhere.")
    recruiting_only: bool = Field(description="true to return only trials that are currently recruiting.")
    max_results: int = Field(description=f"How many trials to return, 1 to {MAX_PAGE_SIZE}.")


class GetCriteriaInput(BaseModel):
    nct_id: str = Field(description="NCT number, e.g. NCT04035486.")


def trial_tools(settings: Settings, seen: dict[str, dict]) -> list[Tool]:
    """``seen`` collects every trial a tool returned, keyed by NCT id, so the caller can
    reject any trial id the model reports without having looked it up."""

    def search_trials(args: SearchTrialsInput) -> dict:
        params = {
            "query.cond": args.condition,
            "pageSize": max(1, min(MAX_PAGE_SIZE, args.max_results)),
            "countTotal": "true",
            "fields": SEARCH_FIELDS,
        }
        if args.other_terms:
            params["query.term"] = args.other_terms
        if args.country:
            params["query.locn"] = args.country
        if args.recruiting_only:
            params["filter.overallStatus"] = "RECRUITING"
        response = httpx.get(SEARCH_URL, params=params, timeout=30, follow_redirects=True)
        if response.status_code == 400:
            raise ToolError("ClinicalTrials.gov rejected the query; simplify the search terms.")
        response.raise_for_status()
        payload = response.json()
        trials = [summarize_study(study) for study in payload.get("studies", [])]
        for trial in trials:
            seen.setdefault(trial["nct_id"], trial)
        return {"total_matching": payload.get("totalCount"), "returned": len(trials), "trials": trials}

    def get_trial_criteria(args: GetCriteriaInput) -> dict:
        try:
            trial = fetch_trial(args.nct_id, settings)
        except (TrialNotFound, ValueError) as exc:
            raise ToolError(str(exc)) from None
        except httpx.HTTPStatusError as exc:
            # The registry answers 400 rather than 404 for some ids it does not hold.
            if exc.response.status_code in (400, 404):
                raise ToolError(f"{args.nct_id} was not found on ClinicalTrials.gov") from None
            raise
        seen.setdefault(trial.nct_id, {"nct_id": trial.nct_id, "title": trial.title, "status": trial.overall_status})
        return {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "status": trial.overall_status,
            "minimum_age": trial.minimum_age,
            "maximum_age": trial.maximum_age,
            "sex": trial.sex,
            "eligibility_criteria": trial.criteria_text,
        }

    return [
        Tool(
            "search_trials",
            "Search the ClinicalTrials.gov registry. The query leaves this system, so send clinical terms "
            "only: never a name, date, record number or any other detail that could identify a person. "
            "Results are summaries; call get_trial_criteria to read a trial's eligibility criteria.",
            SearchTrialsInput,
            search_trials,
        ),
        Tool(
            "get_trial_criteria",
            "Fetch one trial's current eligibility criteria text and age/sex limits from ClinicalTrials.gov.",
            GetCriteriaInput,
            get_trial_criteria,
        ),
    ]
