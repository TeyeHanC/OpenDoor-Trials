"""Command line entry point: python -m trialmatch <command>."""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .models import VERDICT_LABELS, RunResult

_EFFECT_MARK = {"supports": "+", "blocks": "x", "unresolved": "?"}


def print_run(run: RunResult) -> None:
    s = run.summary
    print(f"\n{run.patient.id}  vs  {run.trial.nct_id}  {run.trial.title}")
    print(f"Trial criteria: {run.trial.source} ({run.trial.fetched_at}) | note given to model: {run.note_coverage}")
    print(f"Overall: {s.overall_label}")
    print(
        f"Criteria: {s.counts['total']} total | {s.counts['supports']} support enrollment | "
        f"{s.counts['blocks']} block | {s.counts['unresolved']} unresolved | "
        f"{s.counts['low_confidence']} low confidence"
    )
    stats = run.citation_stats
    print(
        f"Citations: {stats.quotes_verified}/{stats.quotes_total} quotes found verbatim in the note; "
        f"{stats.judgments_downgraded} judgment(s) downgraded to 'insufficient information'"
    )
    print("\n  [+] supports enrollment   [x] blocks enrollment   [?] unresolved\n")
    for c in run.criteria:
        flags = f"  !{','.join(c.flags)}" if c.flags else ""
        if c.basis == "inferred":
            flags += "  (inferred, not directly stated)"
        print(f"[{_EFFECT_MARK[c.effect]}] {c.criterion_id} ({c.type}) {c.criterion_text}")
        print(f"      {VERDICT_LABELS[c.verdict]} | confidence {c.confidence}{flags}")
        print(f"      {c.rationale}")
        if c.patient_value:
            print(f"      patient value: {c.patient_value}")
        for e in c.evidence:
            print(f"      \"{e.quote}\" [{e.start}:{e.end}]")
    if run.tool_calls:
        print(f"\nCalculations and look-ups ({len(run.tool_calls)}):")
        for call in run.tool_calls:
            print(f"  {call.tool}({call.input}) -> {call.output}")
    if s.key_reasons:
        print("\nKey reasons:")
        for item in s.key_reasons:
            print(f"  - {item.text}")
    if s.follow_up:
        print(f"\nTo check ({len(s.follow_up)}):")
        for item in s.follow_up:
            print(f"  - {item.text}")
    print(f"\n{s.disclaimer}")


def cmd_fetch_trec(args: argparse.Namespace) -> int:
    from .patients import fetch_trec, load_qrels, load_trec_topics

    settings = get_settings()
    fetch_trec(settings, args.year, refresh=args.refresh)
    print(
        f"TREC CT {args.year}: {len(load_trec_topics(settings, args.year))} patients, "
        f"{len(load_qrels(settings, args.year))} judged patient-trial pairs in {settings.trec_dir}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .llm import AnthropicLLM
    from .patients import load_patient
    from .pipeline import run_match
    from .store import save_run
    from .trials import get_trial, run_id_suffix

    settings = get_settings()
    patient = load_patient(args.patient, settings)
    trial = get_trial(args.nct, settings, args.trial_source)
    run = run_match(
        patient, trial, settings, AnthropicLLM(settings),
        progress=lambda m: print(m, file=sys.stderr), run_suffix=run_id_suffix(args.trial_source),
    )
    save_run(settings, run)
    print_run(run)
    print(f"\nSaved to {settings.runs_dir / (run.run_id + '.json')}")
    print(f"Review in the browser: python -m trialmatch serve  ->  http://127.0.0.1:8000/?run={run.run_id}")
    return 0


def cmd_find_trials(args: argparse.Namespace) -> int:
    from .find import find_trials, format_find_result, save_find_result
    from .llm import AnthropicLLM
    from .patients import load_patient
    from .pipeline import run_match
    from .store import save_run
    from .trials import get_trial

    settings = get_settings()
    patient = load_patient(args.patient, settings)
    llm = AnthropicLLM(settings)
    result = find_trials(patient, settings, llm, max_candidates=args.max, country=args.country)
    path = save_find_result(settings, result)
    print(format_find_result(result))
    print(f"\nSaved to {path}")

    if args.match:
        print(f"\nMatching {patient.id} against {len(result.candidates)} shortlisted trial(s)")
        for candidate in result.candidates:
            run = run_match(patient, get_trial(candidate.nct_id, settings), settings, llm)
            save_run(settings, run)
            counts = run.summary.counts
            print(
                f"  {candidate.nct_id}  {run.summary.overall_label}  "
                f"({counts['blocks']} blocking, {counts['unresolved']} unresolved of {counts['total']})  ?run={run.run_id}"
            )
    else:
        print("Add --match to run the criterion-level match on each shortlisted trial.")
    return 0


def cmd_compare_batching(args: argparse.Namespace) -> int:
    from .compare import compare_batching, format_diff
    from .llm import AnthropicLLM
    from .patients import load_patient
    from .trials import get_trial, run_id_suffix

    settings = get_settings()
    patient = load_patient(args.patient, settings)
    trial = get_trial(args.nct, settings, args.trial_source)
    diff = compare_batching(
        patient, trial, settings, AnthropicLLM(settings),
        batch_size=args.batch_size, run_suffix=run_id_suffix(args.trial_source),
    )
    print(format_diff(diff))
    print(f"Saved to {diff['output_path']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run("trialmatch.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluation

    return run_evaluation(args)


def cmd_review_stats(args: argparse.Namespace) -> int:
    from .evaluate import print_review_stats

    return print_review_stats(get_settings())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trialmatch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-trec", help="Download TREC Clinical Trials topics and relevance labels")
    p.add_argument("--year", type=int, default=2021, choices=(2021, 2022))
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_fetch_trec)

    p = sub.add_parser("run", help="Match one patient against one trial")
    p.add_argument("--patient", required=True, help="sample-nsclc, trec2021-1, ...")
    p.add_argument("--nct", required=True, help="NCT number, e.g. NCT04035486")
    p.add_argument("--trial-source", choices=("api", "snapshot"), default="api",
                   help="api: current ClinicalTrials.gov record (default); snapshot: TREC 2021-04-27 registry snapshot")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("find-trials", help="Search ClinicalTrials.gov for trials a patient might qualify for")
    p.add_argument("--patient", required=True, help="sample-nsclc, trec2021-1, ...")
    p.add_argument("--max", type=int, default=5, help="Maximum number of trials to shortlist")
    p.add_argument("--country", default=None, help="Restrict to trials with sites in this country")
    p.add_argument("--match", action="store_true", help="Also run the criterion-level match on each shortlisted trial")
    p.set_defaults(func=cmd_find_trials)

    p = sub.add_parser(
        "compare-batching",
        help="Judge one pair twice (several criteria per request vs one per request) and list the disagreements",
    )
    p.add_argument("--patient", required=True)
    p.add_argument("--nct", required=True)
    p.add_argument("--trial-source", choices=("api", "snapshot"), default="api")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size to compare against one-at-a-time")
    p.set_defaults(func=cmd_compare_batching)

    p = sub.add_parser("serve", help="Start the review web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="Evaluate on a sample of TREC-labelled patient-trial pairs")
    p.add_argument("--year", type=int, default=2021, choices=(2021, 2022))
    p.add_argument("--n", type=int, default=500, help="Number of pairs to sample")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--sampling", choices=("balanced", "proportional"), default="balanced")
    p.add_argument("--trial-source", choices=("api", "snapshot"), default="snapshot",
                   help="snapshot (default): criteria as TREC assessors saw them; api: current registry record")
    p.add_argument("--workers", type=int, default=4, help="Pairs processed concurrently")
    p.add_argument("--name", default=None, help="Output folder name under data/eval/")
    p.add_argument("--rerun", action="store_true", help="Ignore saved runs and call the model again")
    p.add_argument("--dry-run", action="store_true", help="Show the sample and exit without calling the model")
    p.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("review-stats", help="Agreement between the model and saved clinician reviews")
    p.set_defaults(func=cmd_review_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .ctgov import TrialNotFound
    from .llm import LLMError
    from .patients import PatientNotFound

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, PatientNotFound, TrialNotFound, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
