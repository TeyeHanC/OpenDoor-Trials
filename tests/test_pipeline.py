from conftest import NOTE, FakeLLM, judgment

from trialmatch.models import effect_of
from trialmatch.pipeline import run_match
from trialmatch.store import list_runs, load_run, save_run
from trialmatch.verify import locate_quote

# Criterion ids produced for the fixture trial:
#   INC-01 Sex is female (registry field; the registry minimum age is dropped
#          because the free text already states an age limit)
#   INC-02 Age >= 18, INC-03 ECOG 0-1, EXC-01 ILD, EXC-02 hepatitis B
GOOD = {
    "INC-01": judgment("INC-01", "met", ["62-year-old woman"]),
    "INC-02": judgment("INC-02", "met", ["62-year-old"], patient_value="62 years"),
    "INC-03": judgment("INC-03", "met", ["ECOG performance status 1"]),
    "EXC-01": judgment("EXC-01", "not_met", ["She denies any history of interstitial lung disease"], segment="S3"),
    "EXC-02": judgment("EXC-02", "unknown", missing_info="Check hepatitis B serology."),
}


def by_id(run):
    return {c.criterion_id: c for c in run.criteria}


def test_locate_quote_is_verbatim_but_whitespace_tolerant():
    assert locate_quote(NOTE, "ECOG performance status 1") is not None
    assert locate_quote("stage IV\nlung cancer", "stage IV lung cancer") == (0, 20)
    assert locate_quote(NOTE, "ecog performance status 1") is None  # case matters
    assert locate_quote(NOTE, "ECOG status 1") is None  # no paraphrase
    assert locate_quote(NOTE, "62-year-old ... adenocarcinoma") is None  # no ellipsis
    assert locate_quote(NOTE, "   ") is None


def test_locate_quote_prefers_cited_segment():
    text = "pain today. No pain yesterday."
    assert locate_quote(text, "pain", within=(12, 30)) == (15, 19)
    assert locate_quote(text, "pain") == (0, 4)


def test_effect_inverts_for_exclusion():
    assert effect_of("inclusion", "met") == "supports"
    assert effect_of("inclusion", "not_met") == "blocks"
    assert effect_of("exclusion", "met") == "blocks"
    assert effect_of("exclusion", "not_met") == "supports"
    assert effect_of("exclusion", "unknown") == "unresolved"


def test_full_run_offsets_summary_and_ids(settings, patient, trial):
    run = run_match(patient, trial, settings, FakeLLM(GOOD))
    results = by_id(run)
    assert list(results) == ["INC-01", "INC-02", "INC-03", "EXC-01", "EXC-02"]
    for c in run.criteria:
        for e in c.evidence:
            assert patient.text[e.start : e.end] == e.quote
    assert results["EXC-01"].verdict == "not_met" and results["EXC-01"].effect == "supports"
    assert results["EXC-02"].effect == "unresolved"
    assert run.summary.overall == "needs_more_info"
    assert [f.criterion_id for f in run.summary.follow_up] == ["EXC-02"]
    assert "hepatitis B serology" in run.summary.follow_up[0].text
    assert run.citation_stats.quotes_total == run.citation_stats.quotes_verified == 4
    assert run.citation_stats.judgments_downgraded == 0


def test_unverifiable_quote_downgrades_to_unknown(settings, patient, trial):
    scripted = dict(GOOD)
    # Family history misread as the patient's own, with a quote that is not in the note.
    scripted["EXC-02"] = judgment("EXC-02", "met", ["patient has hepatitis B"])
    run = run_match(patient, trial, settings, FakeLLM(scripted))
    c = by_id(run)["EXC-02"]
    assert c.model_verdict == "met" and c.verdict == "unknown" and c.downgraded
    assert c.effect == "unresolved" and c.confidence == "low"
    assert c.rejected_quotes == ["patient has hepatitis B"]
    assert "downgraded_no_verified_evidence" in c.flags
    assert run.summary.overall == "needs_more_info"  # not "not_eligible"
    assert run.citation_stats.quotes_verified == 4 and run.citation_stats.quotes_total == 5
    assert run.citation_stats.judgments_downgraded == 1


def test_determination_without_any_quote_is_downgraded(settings, patient, trial):
    scripted = dict(GOOD, **{"INC-03": judgment("INC-03", "met", [])})
    run = run_match(patient, trial, settings, FakeLLM(scripted))
    assert by_id(run)["INC-03"].verdict == "unknown" and by_id(run)["INC-03"].downgraded


def test_met_exclusion_blocks_and_all_clear_is_possibly_eligible(settings, patient, trial):
    blocked = dict(GOOD, **{"EXC-02": judgment("EXC-02", "met", ["ALT 28 U/L"], segment="S5", rationale="why")})
    run = run_match(patient, trial, settings, FakeLLM(blocked))
    assert run.summary.overall == "not_eligible"
    assert run.summary.key_reasons[0].criterion_id == "EXC-02"
    assert run.summary.key_reasons[0].text.startswith("Exclusion criterion applies")

    clear = dict(GOOD, **{"EXC-02": judgment("EXC-02", "not_met", ["ALT 28 U/L"], confidence="low")})
    run = run_match(patient, trial, settings, FakeLLM(clear))
    assert run.summary.overall == "possibly_eligible"
    # Low-confidence determinations are still surfaced for the clinician.
    assert [f.criterion_id for f in run.summary.follow_up] == ["EXC-02"]


def test_missing_judgment_and_refusal_become_flagged_unknowns(settings, patient, trial):
    partial = {k: v for k, v in GOOD.items() if k != "INC-03"}
    run = run_match(patient, trial, settings, FakeLLM(partial))
    assert by_id(run)["INC-03"].flags == ["no_judgment_returned"]

    run = run_match(patient, trial, settings, FakeLLM(GOOD, refuse_judging=True))
    assert all(c.verdict == "unknown" and c.flags == ["model_declined"] for c in run.criteria)


def test_decomposition_is_cached_and_source_text_checked(settings, patient, trial):
    llm = FakeLLM(GOOD)
    run_match(patient, trial, settings, llm)
    first = len(llm.calls)
    run = run_match(patient, trial, settings, llm)
    assert len(llm.calls) == first + 1  # one judging call, no second decomposition
    assert by_id(run)["INC-02"].source_verified
    assert by_id(run)["INC-01"].origin == "structured_field"


def test_run_roundtrip(settings, patient, trial):
    run = run_match(patient, trial, settings, FakeLLM(GOOD))
    save_run(settings, run)
    assert load_run(settings, run.run_id) == run
    assert list_runs(settings)[0]["run_id"] == "sample-test__NCT00000000"
    assert load_run(settings, "missing") is None
