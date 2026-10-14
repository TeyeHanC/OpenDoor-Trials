"""Screen one patient against a pool of trials and rank the results.

This is the entry point a clinician starts from: a note goes in, a ranked list
of trials comes out, and each row opens the criterion-by-criterion match that
produced it (``pipeline.run_match``). A pool is a fixed, hand-picked list of
trials (``trialmatch/pools/*.json``), which keeps a demonstration repeatable;
the trial finder (``find.py``) is the open-ended alternative.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from .config import Settings
from .llm import StructuredLLM
from .models import OVERALL_LABELS, Overall, Patient, RunResult
from .pipeline import make_run_id, run_match
from .store import load_run, save_run
from .trials import get_trial

POOLS_DIR = Path(__file__).parent / "pools"
_POOL_ID_RE = re.compile(r"^[a-z0-9-]+$")
# Best first: nothing rules the patient out, then open questions, then ruled out.
_OVERALL_ORDER = {"possibly_eligible": 0, "needs_more_info": 1, "not_eligible": 2}


class PoolNotFound(Exception):
    pass


class PoolTrial(BaseModel):
    nct_id: str
    title: str = ""
    status_when_selected: Optional[str] = None


class Pool(BaseModel):
    id: str
    title: str
    description: str = ""
    selected_on: Optional[str] = None
    trials: list[PoolTrial]


class ScreenRow(BaseModel):
    nct_id: str
    title: str
    # Registry recruitment status when the trial was fetched, e.g. RECRUITING, ACTIVE_NOT_RECRUITING.
    status: Optional[str] = None
    rank: Optional[int] = None  # None when the trial could not be assessed
    overall: Optional[Overall] = None
    overall_label: str = ""
    counts: dict[str, int] = {}
    headline: str = ""
    run_id: Optional[str] = None
    error: Optional[str] = None


class ScreenResult(BaseModel):
    patient_id: str
    pool_id: str
    pool_title: str
    created_at: str
    model: str
    rows: list[ScreenRow]


def list_pools() -> list[Pool]:
    # "._*" files are macOS metadata that appear next to real files on FAT/exFAT volumes.
    paths = sorted(path for path in POOLS_DIR.glob("*.json") if not path.name.startswith("._"))
    return [Pool.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]


def load_pool(pool_id: str) -> Pool:
    path = POOLS_DIR / f"{pool_id}.json"
    if not _POOL_ID_RE.match(pool_id) or not path.is_file():
        known = ", ".join(pool.id for pool in list_pools()) or "none"
        raise PoolNotFound(f"No trial pool named {pool_id!r} (available: {known})")
    return Pool.model_validate_json(path.read_text(encoding="utf-8"))


def _headline(run: RunResult) -> str:
    counts = run.summary.counts
    if run.summary.overall == "not_eligible":
        return run.summary.key_reasons[0].text if run.summary.key_reasons else "A criterion rules the patient out."
    record = counts.get("missing_from_record", counts["unresolved"])
    parts = [f"{counts['supports']} of {counts['total']} criteria confirmed", "none against"]
    if record:
        parts.append(f"{record} to check in the record")
    if counts.get("confirm_at_screening"):
        parts.append(f"{counts['confirm_at_screening']} to confirm at the screening visit")
    return ", ".join(parts)


def row_for(run: RunResult, title: str = "") -> ScreenRow:
    return ScreenRow(
        nct_id=run.trial.nct_id,
        title=run.trial.title or title,
        status=run.trial.overall_status,
        overall=run.summary.overall,
        overall_label=OVERALL_LABELS[run.summary.overall],
        counts=run.summary.counts,
        headline=_headline(run),
        run_id=run.run_id,
    )


def rank_rows(rows: list[ScreenRow]) -> list[ScreenRow]:
    """Order rows best match first and number them. Trials that failed go last, unranked.

    Within one conclusion: fewer blocking criteria, then fewer open questions the
    record could answer, then more criteria confirmed.
    """

    def key(row: ScreenRow):
        counts = row.counts
        return (
            _OVERALL_ORDER[row.overall],
            counts.get("blocks", 0),
            counts.get("missing_from_record", counts.get("unresolved", 0)),
            -counts.get("supports", 0),
            row.nct_id,
        )

    assessed = sorted((r for r in rows if r.overall is not None), key=key)
    failed = sorted((r for r in rows if r.overall is None), key=lambda r: r.nct_id)
    for position, row in enumerate(assessed, start=1):
        row.rank = position
    for row in failed:
        row.rank = None
    return assessed + failed


def screen_patient(
    patient: Patient,
    pool: Pool,
    settings: Settings,
    llm: StructuredLLM,
    *,
    rerun: bool = False,
    workers: int = 2,
    on_row: Callable[[ScreenRow], None] = lambda row: None,
) -> ScreenResult:
    """Match the patient against every trial in the pool; a saved match is reused unless ``rerun``."""

    def one(entry: PoolTrial) -> ScreenRow:
        try:
            run = None if rerun else load_run(settings, make_run_id(patient.id, entry.nct_id))
            if run is None:
                run = run_match(patient, get_trial(entry.nct_id, settings), settings, llm)
                save_run(settings, run)
            row = row_for(run, entry.title)
        except Exception as exc:  # one trial that cannot be assessed must not lose the others
            row = ScreenRow(
                nct_id=entry.nct_id, title=entry.title, status=entry.status_when_selected,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        on_row(row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one, pool.trials))
    return ScreenResult(
        patient_id=patient.id,
        pool_id=pool.id,
        pool_title=pool.title,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=settings.model,
        rows=rank_rows(rows),
    )


def _screen_path(settings: Settings, patient_id: str, pool_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_.-]+$", patient_id) or not _POOL_ID_RE.match(pool_id):
        raise ValueError("Invalid patient or pool id")
    return settings.data_dir / "screens" / f"{patient_id}__{pool_id}.json"


def save_screen(settings: Settings, result: ScreenResult) -> Path:
    path = _screen_path(settings, result.patient_id, result.pool_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_screen(settings: Settings, patient_id: str, pool_id: str) -> ScreenResult | None:
    path = _screen_path(settings, patient_id, pool_id)
    return ScreenResult.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def _status_note(status: str | None) -> str:
    if not status or status == "RECRUITING":
        return ""
    return f"   [registry status: {status.replace('_', ' ').lower()}]"


def format_screen(result: ScreenResult) -> str:
    assessed = [r for r in result.rows if r.overall]
    lines = [f"{result.patient_id} against {result.pool_title}: {len(assessed)} of {len(result.rows)} trials assessed ({result.model})", ""]
    for row in result.rows:
        if row.overall is None:
            lines += [f"  -  {row.nct_id}  could not be assessed: {row.error}", f"     {row.title}", ""]
            continue
        lines += [
            f"  {row.rank}. {row.nct_id}  {row.overall_label}" + _status_note(row.status),
            f"     {row.title}",
            f"     {row.headline}",
            f"     details: ?run={row.run_id}",
            "",
        ]
    lines.append("A ranking for review, not an enrollment decision; open a trial to see every criterion and its evidence.")
    return "\n".join(lines)
