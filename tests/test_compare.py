from conftest import LLMDecomposition, LLMJudgmentBatch, DECOMPOSITION, judgment

from trialmatch.compare import compare_batching, format_diff
from trialmatch.store import load_run


class BatchSensitiveLLM:
    """Answers EXC-02 differently depending on how many criteria share the request."""

    def structured(self, *, system, user_blocks, schema):
        if schema is LLMDecomposition:
            return LLMDecomposition.model_validate(DECOMPOSITION)
        ids = [cid for cid in ("INC-01", "INC-02", "INC-03", "EXC-01", "EXC-02") if f'"{cid}"' in user_blocks[-1]]
        alone = len(ids) == 1
        out = []
        for cid in ids:
            if cid == "EXC-02":
                out.append(judgment(cid, "unknown") if alone else judgment(cid, "not_met", ["ALT 28 U/L"], segment="S5"))
            else:
                verdict = "not_met" if cid == "EXC-01" else "met"
                out.append(judgment(cid, verdict, ["62-year-old"], confidence="high" if alone or cid != "INC-03" else "medium"))
        return LLMJudgmentBatch.model_validate({"judgments": out})


def test_compare_batching_lists_disagreements_and_saves_both_runs(settings, patient, trial):
    diff = compare_batching(patient, trial, settings, BatchSensitiveLLM(), batch_size=8)
    assert [d["criterion_id"] for d in diff["verdict_disagreements"]] == ["EXC-02"]
    d = diff["verdict_disagreements"][0]
    assert d["batch8"]["verdict"] == "not_met" and d["single"]["verdict"] == "unknown"
    assert d["batch8"]["quotes"] == ["ALT 28 U/L"]
    assert [c["criterion_id"] for c in diff["confidence_only_differences"]] == ["INC-03"]
    assert diff["overall"] == {"batch8": "possibly_eligible", "single": "needs_more_info"}
    assert load_run(settings, "sample-test__NCT00000000__batch8").judge_batch_size == 8
    assert load_run(settings, "sample-test__NCT00000000__single").judge_batch_size == 1
    assert "EXC-02" in format_diff(diff)
