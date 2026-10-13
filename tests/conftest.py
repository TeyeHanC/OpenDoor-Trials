"""Shared fixtures. The model is always replaced by a scripted stub: tests never call the API."""

from __future__ import annotations

import pytest

from trialmatch.config import Settings
from trialmatch.llm import LLMRefusal
from trialmatch.models import LLMDecomposition, LLMJudgmentBatch, Patient, Trial

NOTE = (
    "Note date: 2026-09-18\n"
    "62-year-old woman with stage IV lung adenocarcinoma. ECOG performance status 1.\n"
    "She denies any history of interstitial lung disease.\n"
    "Family history: Mother died of breast cancer at age 71.\n"
    "Labs: ALT 28 U/L."
)

DECOMPOSITION = {
    "criteria": [
        {"type": "inclusion", "text": "Age 18 years or older", "source_text": "Age >= 18 and ECOG 0-1",
         "time_window": None, "threshold": {"parameter": "age", "comparator": ">=", "value": "18", "unit": "years"}, "concepts": []},
        {"type": "inclusion", "text": "ECOG performance status 0-1", "source_text": "Age >= 18 and ECOG 0-1",
         "time_window": None, "threshold": None, "concepts": []},
        {"type": "exclusion", "text": "History of interstitial lung disease", "source_text": "History of ILD",
         "time_window": None, "threshold": None,
         "concepts": [{"original": "ILD", "normalized": "interstitial lung disease", "category": "condition"}]},
        {"type": "exclusion", "text": "Active hepatitis B infection", "source_text": "Active hepatitis B",
         "time_window": None, "threshold": None, "concepts": []},
    ]
}


def judgment(cid, verdict, quotes=(), confidence="high", segment="S2", **extra):
    return {
        "criterion_id": cid,
        "verdict": verdict,
        "evidence": [{"segment_id": segment, "quote": q} for q in quotes],
        "patient_value": extra.get("patient_value"),
        "rationale": extra.get("rationale", "r"),
        "confidence": confidence,
        "missing_info": extra.get("missing_info"),
    }


class FakeLLM:
    """Returns a scripted decomposition and scripted judgments keyed by criterion id."""

    def __init__(self, judgments: dict[str, dict], refuse_judging: bool = False):
        self.judgments = judgments
        self.refuse_judging = refuse_judging
        self.calls: list[dict] = []

    def structured(self, *, system, user_blocks, schema):
        self.calls.append({"schema": schema, "user_blocks": user_blocks})
        if schema is LLMDecomposition:
            return LLMDecomposition.model_validate(DECOMPOSITION)
        if self.refuse_judging:
            raise LLMRefusal("declined")
        asked = [cid for cid in self.judgments if f'"{cid}"' in user_blocks[-1]]
        return LLMJudgmentBatch.model_validate({"judgments": [self.judgments[cid] for cid in asked]})


@pytest.fixture
def settings(tmp_path):
    return Settings(model="test-model", effort="high", data_dir=tmp_path)


@pytest.fixture
def patient():
    return Patient(id="sample-test", source="synthetic", text=NOTE, note_date="2026-09-18")


@pytest.fixture
def trial():
    return Trial(
        nct_id="NCT00000000",
        title="Test trial",
        criteria_text="Inclusion Criteria:\n* Age >= 18 and ECOG 0-1\nExclusion Criteria:\n* History of ILD\n* Active hepatitis B",
        minimum_age="18 Years",
        sex="FEMALE",
        url="https://clinicaltrials.gov/study/NCT00000000",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
