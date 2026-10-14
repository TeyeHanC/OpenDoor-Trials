import functools
import time

import pytest
from conftest import FakeLLM, judgment
from fastapi.testclient import TestClient

from trialmatch import screen as screen_module
from trialmatch import server
from trialmatch.models import Trial
from trialmatch.screen import Pool, PoolTrial, ScreenRow, format_screen, list_pools, load_pool, load_screen, rank_rows, save_screen, screen_patient
from trialmatch.store import load_run

MET = {
    "INC-01": judgment("INC-01", "met", ["62-year-old woman"]),
    "INC-02": judgment("INC-02", "met", ["62-year-old"]),
    "INC-03": judgment("INC-03", "met", ["ECOG performance status 1"]),
    "EXC-01": judgment("EXC-01", "not_met", ["She denies any history of interstitial lung disease"], segment="S3"),
}
CLEAN = dict(MET, **{"EXC-02": judgment("EXC-02", "not_met", ["ALT 28 U/L"], segment="S5")})
OPEN = dict(MET, **{"EXC-02": judgment("EXC-02", "unknown", missing_info="Check hepatitis B serology.")})
BLOCKED = dict(OPEN, **{"INC-03": judgment("INC-03", "not_met", ["ECOG performance status 1"])})

POOL = Pool(id="test-pool", title="Test pool", trials=[
    PoolTrial(nct_id="NCT00000001", title="Blocked trial"),
    PoolTrial(nct_id="NCT00000002", title="Clean trial"),
    PoolTrial(nct_id="NCT00000003", title="Open trial"),
    PoolTrial(nct_id="NCT00000004", title="Broken trial"),
])
SCRIPTS = {"NCT00000001": BLOCKED, "NCT00000002": CLEAN, "NCT00000003": OPEN}


class PoolLLM:
    """Answers each trial from its own script, set when that trial is fetched. This relies on
    trials being matched one at a time, so every test screens with a single worker."""

    def __init__(self):
        self.judged = []
        self.script = {}

    def structured(self, *, system, user_blocks, schema, tools=None, trace=None):
        return FakeLLM(self.script).structured(system=system, user_blocks=user_blocks, schema=schema, tools=tools, trace=trace)


@pytest.fixture
def pool_run(settings, patient, trial, monkeypatch):
    llm = PoolLLM()

    def fake_get_trial(nct_id, _settings, source="api"):
        if nct_id == "NCT00000004":
            raise RuntimeError("registry unavailable")
        llm.script = SCRIPTS[nct_id]
        llm.judged.append(nct_id)
        # A distinct criteria text per trial keeps their cached decompositions apart.
        return trial.model_copy(update={"nct_id": nct_id, "title": f"Title of {nct_id}", "overall_status": "RECRUITING",
                                        "criteria_text": trial.criteria_text + f"\n({nct_id})"})

    monkeypatch.setattr(screen_module, "get_trial", fake_get_trial)
    return llm


def test_bundled_pool_loads_and_unknown_pool_is_reported():
    pool = load_pool("lung-cancer-demo")
    assert len(pool.trials) == 8 and pool.trials[0].nct_id == "NCT04035486"
    assert [p.id for p in list_pools()] == ["lung-cancer-demo"]
    with pytest.raises(screen_module.PoolNotFound, match="available: lung-cancer-demo"):
        load_pool("no-such-pool")
    with pytest.raises(screen_module.PoolNotFound):
        load_pool("../secrets")


def test_trials_are_ranked_best_first_and_a_failed_trial_does_not_lose_the_rest(settings, patient, pool_run):
    result = screen_patient(patient, POOL, settings, pool_run, workers=1)
    assert [(r.nct_id, r.rank, r.overall) for r in result.rows] == [
        ("NCT00000002", 1, "possibly_eligible"),
        ("NCT00000003", 2, "needs_more_info"),
        ("NCT00000001", 3, "not_eligible"),
        ("NCT00000004", None, None),
    ]
    clean, open_, blocked, broken = result.rows
    assert clean.headline == "5 of 5 criteria confirmed, none against"
    assert open_.headline == "4 of 5 criteria confirmed, none against, 1 to check in the record"
    assert blocked.headline.startswith("Inclusion criterion not met: ECOG performance status 0-1")
    assert broken.error == "RuntimeError: registry unavailable" and broken.title == "Broken trial"
    assert clean.run_id == "sample-test__NCT00000002" and load_run(settings, clean.run_id).summary.overall == "possibly_eligible"

    text = format_screen(result)
    assert "1. NCT00000002  Possibly eligible" in text and "could not be assessed: RuntimeError" in text
    assert text.index("NCT00000002") < text.index("NCT00000003") < text.index("NCT00000001")


def test_saved_matches_are_reused_unless_rerun(settings, patient, pool_run):
    screen_patient(patient, POOL, settings, pool_run, workers=1)
    pool_run.judged.clear()
    screen_patient(patient, POOL, settings, pool_run, workers=1)
    assert pool_run.judged == []  # nothing fetched or judged again; only the broken trial was retried and failed
    screen_patient(patient, POOL, settings, pool_run, workers=1, rerun=True)
    assert pool_run.judged == ["NCT00000001", "NCT00000002", "NCT00000003"]


def test_ranking_within_one_conclusion_prefers_fewer_open_questions():
    def row(nct, overall, supports, missing, blocks=0):
        return ScreenRow(nct_id=nct, title=nct, overall=overall,
                         counts={"supports": supports, "blocks": blocks, "missing_from_record": missing, "unresolved": missing})

    ranked = rank_rows([
        row("NCT3", "needs_more_info", supports=9, missing=6), row("NCT1", "needs_more_info", supports=5, missing=2),
        row("NCT2", "needs_more_info", supports=8, missing=2), row("NCT5", "not_eligible", 9, 0, blocks=2),
        row("NCT4", "not_eligible", 2, 5, blocks=1),
    ])
    assert [r.nct_id for r in ranked] == ["NCT2", "NCT1", "NCT3", "NCT4", "NCT5"] and [r.rank for r in ranked] == [1, 2, 3, 4, 5]


def test_non_recruiting_status_is_shown(settings, patient, pool_run):
    result = screen_patient(patient, POOL, settings, pool_run, workers=1)
    result.rows[0].status = "ACTIVE_NOT_RECRUITING"
    assert "[registry status: active not recruiting]" in format_screen(result)
    assert "[registry status" not in format_screen(result).split("2. ")[1]


def test_screen_endpoints_report_progress_then_the_saved_ranking(settings, patient, pool_run, monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(server, "load_patient", lambda patient_id, _settings: patient)
    monkeypatch.setattr(server, "load_pool", lambda pool_id: POOL)
    monkeypatch.setattr(server, "make_llm", lambda _settings: pool_run)
    monkeypatch.setattr(server, "screen_patient", functools.partial(screen_patient, workers=1))
    server._screen_jobs.clear()
    client = TestClient(server.app)

    query = {"patient_id": patient.id, "pool_id": POOL.id}
    assert client.get("/api/screens", params=query).json() == {"state": "none", "rows": []}
    assert client.post("/api/screens", json=query).json() == {"state": "running"}
    for _ in range(100):
        data = client.get("/api/screens", params=query).json()
        if data["state"] != "running":
            break
        assert data["total"] == 4 and len(data["rows"]) == data["done"]
        time.sleep(0.05)
    assert data["state"] == "done" and [r["nct_id"] for r in data["rows"]][:3] == ["NCT00000002", "NCT00000003", "NCT00000001"]
    assert load_screen(settings, patient.id, POOL.id).rows[0].rank == 1
    assert client.get("/api/pools").json()[0]["id"] == "lung-cancer-demo"
    assert client.get("/api/screens", params={"patient_id": "../x", "pool_id": "p"}).status_code == 400
