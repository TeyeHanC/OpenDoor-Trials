"""On-disk storage for run results and clinician reviews.

Runs are one JSON file each. Reviews are an append-only JSONL log so that every
clinician decision (including changes of mind) is kept for later evaluation;
the latest entry per (run, criterion) is the current one.
"""

from __future__ import annotations

import json
import re
import threading

from .config import Settings
from .models import Review, RunResult

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_reviews_lock = threading.Lock()


def _run_path(settings: Settings, run_id: str):
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return settings.runs_dir / f"{run_id}.json"


def save_run(settings: Settings, run: RunResult) -> None:
    path = _run_path(settings, run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_run(settings: Settings, run_id: str) -> RunResult | None:
    path = _run_path(settings, run_id)
    if not path.is_file():
        return None
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(settings: Settings) -> list[dict]:
    runs = []
    for path in settings.runs_dir.glob("*.json"):
        if path.name.startswith("._"):
            continue
        run = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": run.run_id,
                "patient_id": run.patient.id,
                "nct_id": run.trial.nct_id,
                "title": run.trial.title,
                "overall": run.summary.overall,
                "overall_label": run.summary.overall_label,
                "created_at": run.created_at,
            }
        )
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def append_review(settings: Settings, review: Review) -> None:
    settings.reviews_path.parent.mkdir(parents=True, exist_ok=True)
    with _reviews_lock, settings.reviews_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(review.model_dump(), ensure_ascii=False) + "\n")


def load_reviews(settings: Settings) -> list[Review]:
    if not settings.reviews_path.is_file():
        return []
    lines = settings.reviews_path.read_text(encoding="utf-8").splitlines()
    return [Review.model_validate_json(line) for line in lines if line.strip()]


def latest_reviews(settings: Settings, run_id: str | None = None) -> dict[tuple[str, str], Review]:
    latest: dict[tuple[str, str], Review] = {}
    for review in load_reviews(settings):
        if run_id is None or review.run_id == run_id:
            latest[(review.run_id, review.criterion_id)] = review
    return latest
