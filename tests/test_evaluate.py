import csv
import json

from conftest import FakeLLM, judgment

from trialmatch.evaluate import compute_metrics, error_type, evaluate_pairs, sample_pairs, write_outputs
from trialmatch.patients import topics_path

QRELS = [(t, f"NCT{t:04d}{i:04d}", i % 3) for t in range(1, 6) for i in range(30)]


def rec(label, overall, **kw):
    base = {"label": label, "overall": overall, "quotes_total": 4, "quotes_verified": 3,
            "judgments_needing_evidence": 4, "judgments_downgraded": 1, "criteria_total": 5, "criteria_unresolved": 1}
    return {**base, **kw}


def test_sampling_is_deterministic_and_balanced():
    a = sample_pairs(QRELS, 30, seed=1, sampling="balanced")
    assert a == sample_pairs(QRELS, 30, seed=1, sampling="balanced")
    assert a != sample_pairs(QRELS, 30, seed=2, sampling="balanced")
    assert [sum(1 for p in a if p[2] == label) for label in (0, 1, 2)] == [10, 10, 10]
    assert len(sample_pairs(QRELS, 31, seed=1, sampling="balanced")) == 31
    assert len(sample_pairs(QRELS, 20, seed=1, sampling="proportional")) == 20


def test_metrics_screen_in_vs_strict():
    records = [
        rec(2, "possibly_eligible"), rec(2, "needs_more_info"), rec(2, "needs_more_info"), rec(2, "not_eligible"),
        rec(1, "not_eligible"), rec(1, "needs_more_info"),
        rec(0, "not_eligible"), rec(0, "not_eligible"), rec(0, "possibly_eligible"),
    ]
    m = compute_metrics(records)
    assert m["screen_in"] == {"recall": 0.75, "precision": 0.6, "tp": 3, "fp": 2, "fn": 1}
    assert m["strict"] == {"recall": 0.25, "precision": 0.5, "tp": 1, "fp": 1, "fn": 3}
    assert m["confusion_trec_label_by_prediction"]["eligible"]["needs_more_info"] == 2
    assert m["excluded_ruled_out"] == {"trec_excluded_pairs": 2, "predicted_not_eligible": 1, "rate": 0.5}
    assert m["needs_more_info"] == {"pairs": 3, "share_of_all_pairs": 0.3333}
    assert m["citations"]["quote_pass_rate"] == 0.75 and m["citations"]["judgment_pass_rate"] == 0.75
    assert compute_metrics([])["screen_in"]["recall"] is None


def test_error_types():
    assert error_type(2, "not_eligible") == "missed_eligible"
    assert error_type(2, "needs_more_info") is None
    assert error_type(1, "needs_more_info") == "not_ruled_out_but_excluded"
    assert error_type(0, "possibly_eligible") == "not_ruled_out_but_not_relevant"
    assert error_type(0, "not_eligible") is None


def test_evaluate_pairs_exports_errors_and_survives_failures(settings, patient, trial):
    path = topics_path(settings, 2021)
    path.parent.mkdir(parents=True)
    path.write_text(f'<topics><topic number="1">{patient.text}</topic></topics>', encoding="utf-8")

    def fetch(nct_id, _settings, source):
        assert source == "snapshot"
        if nct_id == "NCT99999999":
            raise RuntimeError("gone")
        return trial.model_copy(update={"nct_id": nct_id})

    # The scripted model rules the patient out via EXC-02, on a pair TREC labels eligible.
    llm = FakeLLM({
        "INC-01": judgment("INC-01", "met", ["62-year-old woman"]),
        "INC-02": judgment("INC-02", "met", ["62-year-old"]),
        "INC-03": judgment("INC-03", "met", ["ECOG performance status 1"]),
        "EXC-01": judgment("EXC-01", "not_met", ["She denies any history of interstitial lung disease"]),
        "EXC-02": judgment("EXC-02", "met", ["Mother died of breast cancer"], rationale="misread family history"),
    })
    pairs = [(1, "NCT00000001", 2), (1, "NCT00000002", 0), (1, "NCT99999999", 2)]
    evaluated, failures = evaluate_pairs(pairs, 2021, settings, llm, fetch=fetch, workers=2)
    assert len(evaluated) == 2 and [f["nct_id"] for f in failures] == ["NCT99999999"]

    calls = len(llm.calls)
    evaluate_pairs(pairs[:2], 2021, settings, llm, fetch=fetch)  # saved runs are reused
    assert len(llm.calls) == calls

    out = settings.eval_dir / "t"
    metrics = compute_metrics([r for r, _ in evaluated])
    errors = write_outputs(out, evaluated, failures, metrics, "http://x")
    assert [e["error_type"] for e in errors] == ["missed_eligible"]
    assert errors[0]["blocking_criteria"][0]["quotes"] == ["Mother died of breast cancer"]
    rows = list(csv.DictReader((out / "errors.csv").open()))
    assert rows[0]["nct_id"] == "NCT00000001" and "misread family history" in rows[0]["blocking_criteria"]
    assert rows[0]["review_url"] == "http://x/?run=trec2021-1__NCT00000001__snap2021"
    assert json.loads((out / "metrics.json").read_text())["screen_in"]["fn"] == 1
    assert len(list(csv.DictReader((out / "predictions.csv").open()))) == 2
