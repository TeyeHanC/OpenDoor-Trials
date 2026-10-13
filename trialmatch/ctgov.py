"""Fetch trial eligibility criteria from the ClinicalTrials.gov API v2."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import httpx

from .config import Settings
from .models import Trial

API_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
FIELDS = "NCTId,BriefTitle,OverallStatus,Condition,EligibilityModule"
NCT_RE = re.compile(r"^NCT\d{8}$")


class TrialNotFound(Exception):
    pass


def normalize_nct_id(nct_id: str) -> str:
    nct_id = nct_id.strip().upper()
    if not NCT_RE.match(nct_id):
        raise ValueError(f"Not a valid NCT number: {nct_id!r} (expected NCT followed by 8 digits)")
    return nct_id


def parse_study(payload: dict, fetched_at: str) -> Trial:
    protocol = payload.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    elig = protocol.get("eligibilityModule", {})
    nct_id = ident.get("nctId", "")
    return Trial(
        nct_id=nct_id,
        title=ident.get("briefTitle", ""),
        conditions=protocol.get("conditionsModule", {}).get("conditions", []),
        criteria_text=elig.get("eligibilityCriteria", "") or "",
        minimum_age=elig.get("minimumAge"),
        maximum_age=elig.get("maximumAge"),
        sex=elig.get("sex"),
        overall_status=protocol.get("statusModule", {}).get("overallStatus"),
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        fetched_at=fetched_at,
    )


def fetch_trial(nct_id: str, settings: Settings, *, refresh: bool = False) -> Trial:
    """Return the trial, from the local cache when available.

    Note: the API serves the *current* registry record, which may differ from
    the April 2021 snapshot that the TREC relevance judgments were made on.
    """
    nct_id = normalize_nct_id(nct_id)
    cache_path = settings.trials_dir / f"{nct_id}.json"
    if cache_path.is_file() and not refresh:
        return Trial.model_validate_json(cache_path.read_text(encoding="utf-8"))

    response = httpx.get(
        API_URL.format(nct_id=nct_id), params={"fields": FIELDS}, timeout=30, follow_redirects=True
    )
    if response.status_code == 404:
        raise TrialNotFound(f"{nct_id} was not found on ClinicalTrials.gov")
    response.raise_for_status()
    trial = parse_study(response.json(), datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if not trial.criteria_text.strip():
        raise TrialNotFound(f"{nct_id} has no eligibility criteria text")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(trial.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return trial
