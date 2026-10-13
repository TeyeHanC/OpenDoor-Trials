import pytest
from conftest import FakeLLM, judgment
from fastapi.testclient import TestClient

from trialmatch import server
from trialmatch.pipeline import run_match
from trialmatch.store import load_reviews, save_run

JUDGMENTS = {
    "INC-01": judgment("INC-01", "met", ["62-year-old woman"]),
    "INC-02": judgment("INC-02", "met", ["62-year-old"]),
    "INC-03": judgment("INC-03", "met", ["ECOG performance status 1"]),
    "EXC-01": judgment("EXC-01", "not_met", ["She denies any history of interstitial lung disease"]),
    "EXC-02": judgment("EXC-02", "unknown"),
}


@pytest.fixture
def client(settings, patient, trial, monkeypatch):
    save_run(settings, run_match(patient, trial, settings, FakeLLM(JUDGMENTS)))
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    return TestClient(server.app)


RUN = "sample-test__NCT00000000"


def test_pages_and_run_listing(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert [r["run_id"] for r in client.get("/api/runs").json()] == [RUN]
    body = client.get(f"/api/runs/{RUN}").json()
    assert body["reviews"] == {} and len(body["run"]["criteria"]) == 5
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2Fx").status_code in (400, 404)


def test_reviews_are_appended_and_latest_wins(client, settings):
    url = f"/api/runs/{RUN}/reviews"
    agreed = client.post(url, json={"criterion_id": "INC-03", "action": "agree", "reviewer": "Dr A"}).json()
    assert agreed["reviewer_verdict"] == agreed["model_verdict"] == "met"

    changed = client.post(
        url, json={"criterion_id": "EXC-02", "action": "modify", "reviewer_verdict": "not_met", "comment": "HBsAg negative 2026-08"}
    ).json()
    assert changed["model_verdict"] == "unknown" and changed["reviewer_verdict"] == "not_met"
    client.post(url, json={"criterion_id": "EXC-02", "action": "modify", "reviewer_verdict": "met"})

    assert len(load_reviews(settings)) == 3  # nothing is overwritten
    current = client.get(f"/api/runs/{RUN}").json()["reviews"]
    assert current["EXC-02"]["reviewer_verdict"] == "met" and set(current) == {"INC-03", "EXC-02"}
    assert len(client.get("/api/reviews").json()) == 3


def test_review_validation(client):
    url = f"/api/runs/{RUN}/reviews"
    assert client.post(url, json={"criterion_id": "EXC-02", "action": "modify"}).status_code == 400
    assert client.post(url, json={"criterion_id": "ZZZ-99", "action": "agree"}).status_code == 404
    assert client.post(url, json={"criterion_id": "EXC-02", "action": "bogus"}).status_code == 422


def test_new_run_rejects_unknown_patient_and_bad_nct(client):
    assert client.post("/api/runs", json={"patient_id": "mimic-1", "nct_id": "NCT00000102"}).status_code == 400
    assert client.post("/api/runs", json={"patient_id": "sample-nsclc", "nct_id": "12345"}).status_code == 400
