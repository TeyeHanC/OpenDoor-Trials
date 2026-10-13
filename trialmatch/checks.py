"""Arithmetic the program redoes instead of trusting the model's.

For a criterion that turns on a date window or a numeric limit, the model
reports the inputs it read from the note together with its own answer. The
program computes the answer from those inputs and compares. Agreement is
recorded; disagreement sets the determination aside for manual review. No tool
calling is involved, so this works with any model and the calculation always runs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .models import LLMThresholdCheck, LLMTimeCheck
from .tools import (
    ConvertUnitInput,
    ThresholdInput,
    TimeWindowInput,
    ToolError,
    _normalize_unit,
    check_time_window,
    compare_to_threshold,
    convert_lab_unit,
)

CheckStatus = Literal["confirmed", "mismatch", "not_checkable"]


class Check(BaseModel):
    kind: Literal["time_window", "threshold"]
    inputs: dict
    model_answer: bool
    computed: Optional[bool] = None
    status: CheckStatus
    detail: str
    # Inputs the model gave that do not appear in the note as written.
    inputs_not_in_note: list[str] = []


def _number_text(value: float) -> str:
    return f"{value:g}"


def run_time_check(check: LLMTimeCheck, note: str) -> Check:
    inputs = check.model_dump(exclude={"within_window"})
    try:
        out = check_time_window(
            TimeWindowInput(
                event_date=check.event_date, reference_date=check.reference_date,
                amount=check.window_amount, unit=check.window_unit,
            )
        )
    except ToolError as exc:
        return Check(kind="time_window", inputs=inputs, model_answer=check.within_window, status="not_checkable", detail=str(exc))
    computed = out["event_is_within_window"]
    where = "after the reference date" if out["event_is_after_reference"] else f"{out['days_before_reference']} days before {check.reference_date}"
    detail = (
        f"{check.event_date} is {where}; the {check.window_amount}-{check.window_unit[:-1]} window "
        f"runs from {out['window_start']} to {out['window_end']}, so the event is {'inside' if computed else 'outside'} it"
    )
    return Check(
        kind="time_window", inputs=inputs, model_answer=check.within_window, computed=computed,
        status="confirmed" if computed == check.within_window else "mismatch", detail=detail,
        inputs_not_in_note=[d for d in (check.event_date,) if d not in note],
    )


def run_threshold_check(check: LLMThresholdCheck, note: str) -> Check:
    inputs = check.model_dump(exclude={"satisfied"})

    def unchecked(reason: str) -> Check:
        return Check(kind="threshold", inputs=inputs, model_answer=check.satisfied, status="not_checkable", detail=reason)

    value, converted = check.value, ""
    if check.value_unit and check.threshold_unit and check.reference_limit is None:
        if _normalize_unit(check.value_unit) != _normalize_unit(check.threshold_unit):
            if check.analyte is None:
                return unchecked(f"Units differ ({check.value_unit} vs {check.threshold_unit}) and no analyte was given for conversion.")
            try:
                value = convert_lab_unit(
                    ConvertUnitInput(analyte=check.analyte, value=check.value, from_unit=check.value_unit, to_unit=check.threshold_unit)
                )["value"]
            except ToolError as exc:
                return unchecked(str(exc))
            converted = f"{_number_text(check.value)} {check.value_unit} = {_number_text(value)} {check.threshold_unit}; "
    try:
        out = compare_to_threshold(
            ThresholdInput(
                patient_value=value, comparator=check.comparator, threshold=check.threshold,
                threshold_upper=check.threshold_upper, upper_limit_of_normal=check.reference_limit,
            )
        )
    except ToolError as exc:
        return unchecked(str(exc))
    computed = out["satisfied"]
    limit = out.get("effective_threshold", out.get("effective_range"))
    unit = check.threshold_unit or check.value_unit or ""
    scaled = f" ({_number_text(check.threshold)} x {_number_text(check.reference_limit)})" if check.reference_limit is not None else ""
    detail = f"{converted}{_number_text(value)} {check.comparator} {limit}{scaled} {unit}".strip() + f" is {'true' if computed else 'false'}"
    return Check(
        kind="threshold", inputs=inputs, model_answer=check.satisfied, computed=computed,
        status="confirmed" if computed == check.satisfied else "mismatch", detail=detail,
        inputs_not_in_note=[_number_text(check.value)] if _number_text(check.value) not in note else [],
    )


def run_checks(time_check: LLMTimeCheck | None, threshold_check: LLMThresholdCheck | None, note: str) -> list[Check]:
    checks = []
    if time_check is not None:
        checks.append(run_time_check(time_check, note))
    if threshold_check is not None:
        checks.append(run_threshold_check(threshold_check, note))
    return checks
