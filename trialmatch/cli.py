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
        print(f"[{_EFFECT_MARK[c.effect]}] {c.criterion_id} ({c.type}) {c.criterion_text}")
        print(f"      {VERDICT_LABELS[c.verdict]} | confidence {c.confidence}{flags}")
        print(f"      {c.rationale}")
        if c.patient_value:
            print(f"      patient value: {c.patient_value}")
        for e in c.evidence:
            print(f"      \"{e.quote}\" [{e.start}:{e.end}]")
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
    from .ctgov import fetch_trial
    from .llm import AnthropicLLM
    from .patients import load_patient
    from .pipeline import run_match
    from .store import save_run

    settings = get_settings()
    patient = load_patient(args.patient, settings)
    trial = fetch_trial(args.nct, settings)
    run = run_match(patient, trial, settings, AnthropicLLM(settings), progress=lambda m: print(m, file=sys.stderr))
    save_run(settings, run)
    print_run(run)
    print(f"\nSaved to {settings.runs_dir / (run.run_id + '.json')}")
    print(f"Review in the browser: python -m trialmatch serve  ->  http://127.0.0.1:8000/?run={run.run_id}")
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
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="Start the review web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="Evaluate on a sample of TREC-labelled patient-trial pairs")
    p.add_argument("--year", type=int, default=2021, choices=(2021, 2022))
    p.add_argument("--n", type=int, default=500, help="Number of pairs to sample")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--sampling", choices=("balanced", "proportional"), default="balanced")
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
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
