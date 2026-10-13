"""Local review web app: note on the left, criteria on the right."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_settings
from .ctgov import TrialNotFound, fetch_trial
from .llm import LLMError, MissingCredentials, make_llm
from .models import Review, Verdict
from .patients import PatientNotFound, list_patients, load_patient
from .pipeline import run_match
from .store import append_review, latest_reviews, list_runs, load_reviews, load_run, save_run

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Trial eligibility screening")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    patient_id: str
    nct_id: str


class ReviewRequest(BaseModel):
    criterion_id: str
    action: Literal["agree", "modify"]
    reviewer_verdict: Optional[Verdict] = None
    comment: str = ""
    reviewer: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/patients")
def api_patients() -> list[dict]:
    return list_patients(get_settings())


@app.get("/api/runs")
def api_runs() -> list[dict]:
    return list_runs(get_settings())


def _get_run(run_id: str):
    try:
        run = load_run(get_settings(), run_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if run is None:
        raise HTTPException(404, f"No run named {run_id}")
    return run


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    run = _get_run(run_id)
    reviews = latest_reviews(get_settings(), run_id)
    return {"run": run.model_dump(), "reviews": {cid: r.model_dump() for (_, cid), r in reviews.items()}}


@app.post("/api/runs")
def api_create_run(request: RunRequest) -> dict:
    """Run the full pipeline for one patient and one trial (takes a minute or so)."""
    settings = get_settings()
    try:
        patient = load_patient(request.patient_id, settings)
        trial = fetch_trial(request.nct_id, settings)
    except (PatientNotFound, TrialNotFound, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        run = run_match(patient, trial, settings, make_llm(settings))
    except anthropic.AuthenticationError as exc:
        raise HTTPException(400, "The Anthropic API rejected the API key. Check ANTHROPIC_API_KEY in .env.") from exc
    except anthropic.APIError as exc:
        raise HTTPException(502, f"Anthropic API error: {exc}") from exc
    except MissingCredentials as exc:
        raise HTTPException(400, f"{exc} Then restart the server.") from exc
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    save_run(settings, run)
    return {"run_id": run.run_id}


@app.post("/api/runs/{run_id}/reviews")
def api_review(run_id: str, request: ReviewRequest) -> dict:
    run = _get_run(run_id)
    criterion = next((c for c in run.criteria if c.criterion_id == request.criterion_id), None)
    if criterion is None:
        raise HTTPException(404, f"Run {run_id} has no criterion {request.criterion_id}")
    if request.action == "modify" and request.reviewer_verdict is None:
        raise HTTPException(400, "A modified review needs reviewer_verdict")
    review = Review(
        run_id=run_id,
        criterion_id=criterion.criterion_id,
        action=request.action,
        model_verdict=criterion.verdict,
        reviewer_verdict=criterion.verdict if request.action == "agree" else request.reviewer_verdict,
        comment=request.comment.strip(),
        reviewer=request.reviewer.strip(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    append_review(get_settings(), review)
    return review.model_dump()


@app.get("/api/reviews")
def api_reviews() -> list[dict]:
    """Full review log (every entry, oldest first) for export."""
    return [r.model_dump() for r in load_reviews(get_settings())]
