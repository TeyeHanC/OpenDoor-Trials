"""Judge the same patient-trial pair with two batch sizes and list the disagreements.

Used to decide whether judging several criteria per request (cheaper) changes
verdicts compared with judging each criterion on its own. Both runs share one
cached decomposition, so they judge identical criteria. Model output is not
deterministic, so a few differences can appear even between two runs with the
same batch size; treat single disagreements as leads, not proof.
"""

from __future__ import annotations

import dataclasses
import json

from .config import Settings
from .llm import StructuredLLM
from .models import CriterionResult, Patient, RunResult, Trial
from .pipeline import run_match
from .store import save_run


def _side(c: CriterionResult) -> dict:
    return {
        "verdict": c.verdict,
        "effect": c.effect,
        "confidence": c.confidence,
        "basis": c.basis,
        "rationale": c.rationale,
        "quotes": [e.quote for e in c.evidence],
        "downgraded": c.downgraded,
    }


def diff_runs(a: RunResult, b: RunResult, label_a: str, label_b: str) -> dict:
    other = {c.criterion_id: c for c in b.criteria}
    verdict_diffs, confidence_only = [], []
    for ca in a.criteria:
        cb = other[ca.criterion_id]
        if ca.verdict != cb.verdict:
            verdict_diffs.append(
                {"criterion_id": ca.criterion_id, "type": ca.type, "criterion": ca.criterion_text,
                 label_a: _side(ca), label_b: _side(cb)}
            )
        elif ca.confidence != cb.confidence:
            confidence_only.append(
                {"criterion_id": ca.criterion_id, "verdict": ca.verdict, label_a: ca.confidence, label_b: cb.confidence}
            )
    return {
        "patient_id": a.patient.id,
        "nct_id": a.trial.nct_id,
        "trial_source": a.trial.source,
        "criteria_total": len(a.criteria),
        "runs": {label_a: a.run_id, label_b: b.run_id},
        "overall": {label_a: a.summary.overall, label_b: b.summary.overall},
        "verdict_disagreements": verdict_diffs,
        "confidence_only_differences": confidence_only,
    }


def compare_batching(
    patient: Patient, trial: Trial, settings: Settings, llm: StructuredLLM, *, batch_size: int = 8, run_suffix: str = ""
) -> dict:
    """Run once with ``batch_size`` criteria per request and once with one; save both runs and the diff."""
    labels = {batch_size: f"batch{batch_size}", 1: "single"}
    runs = {}
    for size, label in labels.items():
        sized = dataclasses.replace(settings, judge_batch_size=size)
        runs[label] = run_match(patient, trial, sized, llm, run_suffix=f"{run_suffix}__{label}")
        save_run(settings, runs[label])
    label_a, label_b = labels[batch_size], labels[1]
    diff = diff_runs(runs[label_a], runs[label_b], label_a, label_b)

    out = settings.data_dir / "compare" / f"{patient.id}__{trial.nct_id}{run_suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")
    diff["output_path"] = str(out)
    return diff


def format_diff(diff: dict) -> str:
    label_a, label_b = diff["runs"]
    lines = [
        f"{diff['patient_id']} vs {diff['nct_id']} ({diff['trial_source']}): {diff['criteria_total']} criteria",
        f"Overall: {label_a} = {diff['overall'][label_a]}, {label_b} = {diff['overall'][label_b]}",
        f"Verdict disagreements: {len(diff['verdict_disagreements'])}   "
        f"(same verdict, different confidence: {len(diff['confidence_only_differences'])})",
    ]
    for d in diff["verdict_disagreements"]:
        lines.append(f"\n{d['criterion_id']} ({d['type']}) {d['criterion']}")
        for label in (label_a, label_b):
            side = d[label]
            quotes = " | ".join(f'"{q}"' for q in side["quotes"]) or "no quote"
            lines.append(f"  {label:8} {side['verdict']:8} {side['confidence']:6} {side['rationale']}")
            lines.append(f"           {quotes}")
    lines.append(f"\nOpen either run in the review app: ?run={diff['runs'][label_a]}  or  ?run={diff['runs'][label_b]}")
    return "\n".join(lines)
