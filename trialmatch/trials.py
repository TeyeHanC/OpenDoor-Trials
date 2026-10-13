"""Where a trial's eligibility criteria come from.

* ``api``      the current ClinicalTrials.gov record (demo and product path)
* ``snapshot`` the 2021-04-27 registry snapshot TREC assessors judged (evaluation)
"""

from __future__ import annotations

from .config import Settings
from .ctgov import fetch_trial
from .models import Trial
from .trec_trials import fetch_snapshot_trial

TRIAL_SOURCES = ("api", "snapshot")


def get_trial(nct_id: str, settings: Settings, source: str = "api") -> Trial:
    if source == "api":
        return fetch_trial(nct_id, settings)
    if source == "snapshot":
        return fetch_snapshot_trial(nct_id, settings)
    raise ValueError(f"Unknown trial source {source!r}; expected one of {TRIAL_SOURCES}")


def run_id_suffix(source: str) -> str:
    """Snapshot runs get their own id so they never overwrite or reuse a live-registry run."""
    return "" if source == "api" else "__snap2021"
