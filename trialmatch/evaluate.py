"""Evaluation against TREC Clinical Trials relevance labels.

TREC labels each (patient, trial) pair as 0 = not relevant, 1 = excluded,
2 = eligible. Gold positive here means label 2.

The system has three overall outcomes, so "predicted positive" is reported
under two definitions:

* screen-in (headline): the patient was NOT ruled out, i.e. "possibly
  eligible" or "more information needed". This is the recall-oriented reading:
  a patient is only lost when the system says "not eligible".
* strict: only "possibly eligible" counts as positive.

Screen-in recall alone is meaningless (ruling nobody out gives 100%), so it is
always reported next to two counterweights: how many TREC-"excluded" pairs the
system did rule out, and what share of all pairs ended as "more information
needed".

Trial criteria come from the 2021-04-27 registry snapshot TREC assessors saw
(``--trial-source snapshot``, the default), so an error is never just a
criterion that changed in the live registry afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from .config import Settings, get_settings
from .llm import StructuredLLM
from .models import RunResult, Trial
from .patients import load_patient, load_qrels
from .pipeline import make_run_id, run_match
from .store import latest_reviews, load_run, save_run
from .trials import get_trial, run_id_suffix

LABEL_NAMES = {0: "not_relevant", 1: "excluded", 2: "eligible"}
OVERALLS = ("possibly_eligible", "needs_more_info", "not_eligible")
Pair = tuple[int, str, int]  # topic number, NCT id, TREC label


def sample_pairs(qrels: list[Pair], n: int, seed: int, sampling: str) -> list[Pair]:
    """Deterministic sample. "balanced" draws equally from the three labels."""
    rng = random.Random(seed)
    if sampling == "proportional":
        return sorted(rng.sample(qrels, min(n, len(qrels))))
    chosen: list[Pair] = []
    for i, label in enumerate((2, 1, 0)):
        pool = sorted(p for p in qrels if p[2] == label)
        quota = n // 3 + (1 if i < n % 3 else 0)
        chosen.extend(rng.sample(pool, min(quota, len(pool))))
    return sorted(chosen)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _precision_recall(records: list[dict], is_positive: Callable[[str], bool]) -> dict:
    tp = sum(1 for r in records if r["label"] == 2 and is_positive(r["overall"]))
    fn = sum(1 for r in records if r["label"] == 2 and not is_positive(r["overall"]))
    fp = sum(1 for r in records if r["label"] != 2 and is_positive(r["overall"]))
    return {"recall": _ratio(tp, tp + fn), "precision": _ratio(tp, tp + fp), "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(records: list[dict]) -> dict:
    """``records`` are the successfully evaluated pairs (see ``record_for``)."""
    confusion = {
        LABEL_NAMES[label]: {o: sum(1 for r in records if r["label"] == label and r["overall"] == o) for o in OVERALLS}
        for label in (2, 1, 0)
    }
    quotes_total = sum(r["quotes_total"] for r in records)
    quotes_verified = sum(r["quotes_verified"] for r in records)
    needing = sum(r["judgments_needing_evidence"] for r in records)
    downgraded = sum(r["judgments_downgraded"] for r in records)
    excluded = [r for r in records if r["label"] == 1]
    excluded_ruled_out = sum(1 for r in excluded if r["overall"] == "not_eligible")
    needs_info = sum(1 for r in records if r["overall"] == "needs_more_info")
    return {
        "pairs_evaluated": len(records),
        "gold_eligible": sum(1 for r in records if r["label"] == 2),
        "screen_in": _precision_recall(records, lambda o: o != "not_eligible"),
        "strict": _precision_recall(records, lambda o: o == "possibly_eligible"),
        "excluded_ruled_out": {
            "trec_excluded_pairs": len(excluded),
            "predicted_not_eligible": excluded_ruled_out,
            "rate": _ratio(excluded_ruled_out, len(excluded)),
        },
        "needs_more_info": {"pairs": needs_info, "share_of_all_pairs": _ratio(needs_info, len(records))},
        "confusion_trec_label_by_prediction": confusion,
        "citations": {
            "quotes_total": quotes_total,
            "quotes_verified": quotes_verified,
            "quote_pass_rate": _ratio(quotes_verified, quotes_total),
            "judgments_needing_evidence": needing,
            "judgments_downgraded": downgraded,
            "judgment_pass_rate": _ratio(needing - downgraded, needing),
        },
        "criteria": {
            "total": sum(r["criteria_total"] for r in records),
            "unresolved": sum(r["criteria_unresolved"] for r in records),
        },
    }


def record_for(run: RunResult, topic: int, label: int) -> dict:
    return {
        "run_id": run.run_id,
        "topic": topic,
        "nct_id": run.trial.nct_id,
        "label": label,
        "label_name": LABEL_NAMES[label],
        "overall": run.summary.overall,
        "criteria_total": len(run.criteria),
        "criteria_blocking": run.summary.counts["blocks"],
        "criteria_unresolved": run.summary.counts["unresolved"],
        "quotes_total": run.citation_stats.quotes_total,
        "quotes_verified": run.citation_stats.quotes_verified,
        "judgments_needing_evidence": run.citation_stats.judgments_needing_evidence,
        "judgments_downgraded": run.citation_stats.judgments_downgraded,
    }


def error_type(label: int, overall: str) -> str | None:
    """Errors under the screen-in definition; None when the pair is handled correctly."""
    if label == 2 and overall == "not_eligible":
        return "missed_eligible"  # the costly kind: a suitable patient was ruled out
    if label == 1 and overall != "not_eligible":
        return "not_ruled_out_but_excluded"
    if label == 0 and overall != "not_eligible":
        return "not_ruled_out_but_not_relevant"
    return None


def error_detail(run: RunResult, record: dict, kind: str, base_url: str) -> dict:
    def brief(c) -> dict:
        return {
            "criterion_id": c.criterion_id,
            "type": c.type,
            "criterion": c.criterion_text,
            "verdict": c.verdict,
            "confidence": c.confidence,
            "rationale": c.rationale,
            "quotes": [e.quote for e in c.evidence],
        }

    return {
        **record,
        "error_type": kind,
        "review_url": f"{base_url}/?run={run.run_id}",
        "trial_title": run.trial.title,
        "patient_note": run.patient.text,
        "blocking_criteria": [brief(c) for c in run.criteria if c.effect == "blocks"],
        "unresolved_criteria": [brief(c) for c in run.criteria if c.effect == "unresolved"],
    }


def evaluate_pairs(
    pairs: list[Pair],
    year: int,
    settings: Settings,
    llm: StructuredLLM,
    *,
    trial_source: str = "snapshot",
    fetch: Callable[[str, Settings, str], Trial] = get_trial,
    workers: int = 4,
    rerun: bool = False,
    progress: Callable[[str], None] = lambda message: None,
) -> tuple[list[tuple[dict, RunResult]], list[dict]]:
    """Run (or reuse) every pair. Returns (evaluated, failures)."""

    def one(pair: Pair):
        topic, nct_id, label = pair
        patient_id = f"trec{year}-{topic}"
        try:
            suffix = run_id_suffix(trial_source)
            run = None if rerun else load_run(settings, make_run_id(patient_id, nct_id, suffix))
            if run is None:
                trial = fetch(nct_id, settings, trial_source)
                run = run_match(load_patient(patient_id, settings), trial, settings, llm, run_suffix=suffix)
                save_run(settings, run)
            return record_for(run, topic, label), run, None
        except Exception as exc:  # one bad pair must not abort a long evaluation
            return None, None, {"topic": topic, "nct_id": nct_id, "label": label, "error": f"{type(exc).__name__}: {exc}"}

    evaluated, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (record, run, failure) in enumerate(pool.map(one, pairs), start=1):
            if failure:
                failures.append(failure)
            else:
                evaluated.append((record, run))
            if done % 10 == 0 or done == len(pairs):
                progress(f"{done}/{len(pairs)} pairs done, {len(failures)} failed")
    return evaluated, failures


def write_outputs(out_dir, evaluated, failures, metrics: dict, base_url: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record, _ in evaluated]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if records:
        with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    errors = []
    for record, run in evaluated:
        kind = error_type(record["label"], record["overall"])
        if kind:
            errors.append(error_detail(run, record, kind, base_url))
    # Missed eligible patients first: they are the errors that matter most.
    errors.sort(key=lambda e: (e["error_type"] != "missed_eligible", e["topic"], e["nct_id"]))

    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["error_type", "topic", "nct_id", "trec_label", "predicted", "blocking_criteria", "n_unresolved", "review_url", "trial_title"]
        )
        for e in errors:
            blocking = " || ".join(
                f"[{b['criterion_id']} {b['type']} {b['verdict']}, {b['confidence']}] {b['criterion']} -- {b['rationale']} "
                f"-- quotes: {' | '.join(b['quotes'])}"
                for b in e["blocking_criteria"]
            )
            writer.writerow(
                [e["error_type"], e["topic"], e["nct_id"], e["label_name"], e["overall"], blocking,
                 len(e["unresolved_criteria"]), e["review_url"], e["trial_title"]]
            )
    return errors


def format_report(metrics: dict, errors: list[dict], failures: list[dict], out_dir) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    s, t, c = metrics["screen_in"], metrics["strict"], metrics["citations"]
    lines = [
        f"Pairs evaluated: {metrics['pairs_evaluated']} ({metrics['gold_eligible']} TREC-eligible), {len(failures)} failed",
        "",
        "Patient-trial level (gold positive = TREC 'eligible')",
        f"  Screen-in (not ruled out):   recall {pct(s['recall'])}   precision {pct(s['precision'])}   "
        f"[TP {s['tp']}  FP {s['fp']}  FN {s['fn']}]",
        f"  Strict (possibly eligible):  recall {pct(t['recall'])}   precision {pct(t['precision'])}   "
        f"[TP {t['tp']}  FP {t['fp']}  FN {t['fn']}]",
        "",
        "Read screen-in recall together with:",
        f"  TREC-excluded pairs ruled out ('not eligible'):  {metrics['excluded_ruled_out']['predicted_not_eligible']}/"
        f"{metrics['excluded_ruled_out']['trec_excluded_pairs']} ({pct(metrics['excluded_ruled_out']['rate'])})",
        f"  'More information needed' share of all pairs:   {metrics['needs_more_info']['pairs']}/"
        f"{metrics['pairs_evaluated']} ({pct(metrics['needs_more_info']['share_of_all_pairs'])})",
        "",
        "TREC label x predicted conclusion",
        f"  {'':14}" + "".join(f"{o:>20}" for o in OVERALLS),
    ]
    for label_name, row in metrics["confusion_trec_label_by_prediction"].items():
        lines.append(f"  {label_name:14}" + "".join(f"{row[o]:>20}" for o in OVERALLS))
    lines += [
        "",
        f"Citation check: {c['quotes_verified']}/{c['quotes_total']} quotes found verbatim ({pct(c['quote_pass_rate'])}); "
        f"{c['judgments_downgraded']}/{c['judgments_needing_evidence']} determinations downgraded for lack of a verified quote",
        f"Criteria judged: {metrics['criteria']['total']}, of which {metrics['criteria']['unresolved']} unresolved",
        "",
        "Errors exported: " + (", ".join(f"{n} {k}" for k, n in Counter(e["error_type"] for e in errors).items()) or "none"),
        f"  {out_dir / 'errors.csv'}   (one row per case, with a link to open it in the review app)",
        f"  {out_dir / 'errors.jsonl'} (full detail)",
    ]
    return "\n".join(lines)


def run_evaluation(args: argparse.Namespace) -> int:
    from .llm import AnthropicLLM

    settings = get_settings()
    pairs = sample_pairs(load_qrels(settings, args.year), args.n, args.seed, args.sampling)
    by_label = Counter(LABEL_NAMES[label] for _, _, label in pairs)
    suffix = run_id_suffix(args.trial_source)
    cached = sum(1 for topic, nct, _ in pairs if load_run(settings, make_run_id(f"trec{args.year}-{topic}", nct, suffix)))
    to_run = len(pairs) if args.rerun else len(pairs) - cached
    print(
        f"TREC CT {args.year}: {len(pairs)} pairs sampled ({args.sampling}, seed {args.seed}): {dict(by_label)}\n"
        f"Trial criteria source: {args.trial_source}"
        + ("" if args.trial_source == "snapshot" else "  (WARNING: live registry text may differ from what TREC judged)")
        + f"\n{cached} already have a saved run; {to_run} will call {settings.model}."
    )
    if args.sampling == "balanced":
        print("Note: balanced sampling over-represents eligible pairs, so precision is not the real-world rate.")
    if args.dry_run:
        return 0
    if to_run and not args.yes:
        answer = input(f"Run {to_run} pairs now? Each pair makes several model calls. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1

    llm = AnthropicLLM(settings)
    evaluated, failures = evaluate_pairs(
        pairs, args.year, settings, llm, trial_source=args.trial_source, workers=args.workers, rerun=args.rerun, progress=lambda m: print(m, file=sys.stderr)
    )
    metrics = compute_metrics([record for record, _ in evaluated])
    metrics["config"] = {
        "year": args.year, "n": args.n, "seed": args.seed, "sampling": args.sampling, "trial_source": args.trial_source, "model": settings.model,
        "full_note_max_chars": settings.full_note_max_chars, "judge_batch_size": settings.judge_batch_size,
        "effort": settings.effort, "failed_pairs": len(failures), "usage": llm.usage_summary(),
    }
    name = args.name or f"trec{args.year}-n{args.n}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = settings.eval_dir / name
    errors = write_outputs(out_dir, evaluated, failures, metrics, base_url="http://127.0.0.1:8000")
    print("\n" + format_report(metrics, errors, failures, out_dir))
    usage = metrics["config"]["usage"]
    if usage["requests"]:
        print(
            f"\nModel usage this run: {usage['requests']} requests, {usage['input_tokens']:,} input and "
            f"{usage['output_tokens']:,} output tokens"
        )
    if failures:
        print(f"{len(failures)} pairs failed (see {out_dir / 'failures.json'}); re-run the same command to retry them.")
    return 0


def print_review_stats(settings: Settings) -> int:
    """How often clinicians agreed with the system, from the saved review log."""
    reviews = list(latest_reviews(settings).values())
    if not reviews:
        print("No clinician reviews saved yet.")
        return 0
    agreed = sum(1 for r in reviews if r.reviewer_verdict == r.model_verdict)
    print(f"{len(reviews)} criteria reviewed; clinician agreed with the system on {agreed} ({agreed / len(reviews):.1%})\n")
    print("system verdict -> clinician verdict")
    transitions = Counter((r.model_verdict, r.reviewer_verdict) for r in reviews)
    for (system, clinician), count in sorted(transitions.items()):
        print(f"  {system:8} -> {clinician:8} {count}")
    changed = [r for r in reviews if r.reviewer_verdict != r.model_verdict]
    if changed:
        print("\nChanged by clinician:")
        for r in changed:
            print(f"  {r.run_id} {r.criterion_id}: {r.model_verdict} -> {r.reviewer_verdict}  {r.comment}")
    return 0
