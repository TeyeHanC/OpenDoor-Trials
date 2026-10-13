import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import FakeLLM, judgment

from trialmatch import tools as tools_module
from trialmatch.find import LLMShortlist, find_trials
from trialmatch.llm import MAX_TOOL_TURNS, AnthropicLLM, LLMError
from trialmatch.models import LLMJudgmentBatch, Patient
from trialmatch.pipeline import run_match
from trialmatch.segment import segment_note
from trialmatch.tools import calculator_tools, note_tools, trial_tools


def call(tool_list, name, tool_input):
    output, is_error = {t.name: t for t in tool_list}[name].run(tool_input)
    return (output if is_error else json.loads(output)), is_error


# ---- definitions ----------------------------------------------------------


def test_every_tool_definition_is_strict(settings):
    segments = segment_note("One.\nTwo.")
    for tool in calculator_tools() + note_tools(segments) + trial_tools(settings, {}):
        definition = tool.definition()
        schema = definition["input_schema"]
        assert definition["strict"] is True
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])


# ---- calculators ----------------------------------------------------------


def test_time_window_uses_calendar_months_and_reports_days():
    calc = calculator_tools()
    out, _ = call(calc, "check_time_window",
                  {"event_date": "2026-08-04", "reference_date": "2026-09-18", "amount": 28, "unit": "days"})
    assert out["days_before_reference"] == 45 and out["event_is_within_window"] is False
    out, _ = call(calc, "check_time_window",
                  {"event_date": "2026-08-04", "reference_date": "2026-09-18", "amount": 6, "unit": "months"})
    assert out["window_start"] == "2026-03-18" and out["event_is_within_window"] is True
    # Month arithmetic clamps to the end of a shorter month.
    out, _ = call(calc, "check_time_window",
                  {"event_date": "2026-03-01", "reference_date": "2026-03-31", "amount": 1, "unit": "months"})
    assert out["window_start"] == "2026-02-28"
    out, _ = call(calc, "date_difference", {"earlier_date": "2014-06-01", "later_date": "2026-09-18"})
    assert out["days"] == 4492 and out["years"] == 12.3


def test_partial_or_missing_dates_are_errors_not_guesses():
    out, is_error = call(calculator_tools(), "date_difference", {"earlier_date": "2026-08", "later_date": "2026-09-18"})
    assert is_error and "YYYY-MM-DD" in out
    out, is_error = call(calculator_tools(), "date_difference", {"earlier_date": "2026-08-01"})
    assert is_error and "later_date" in out


def test_threshold_comparison_including_multiples_of_uln():
    calc = calculator_tools()
    base = {"threshold_upper": None, "upper_limit_of_normal": None}
    out, _ = call(calc, "compare_to_threshold", {"patient_value": 3.4, "comparator": ">=", "threshold": 1.5, **base})
    assert out == {"effective_threshold": 1.5, "satisfied": True}
    out, _ = call(calc, "compare_to_threshold",
                  {"patient_value": 120, "comparator": "<=", "threshold": 2.5, "threshold_upper": None, "upper_limit_of_normal": 40})
    assert out == {"effective_threshold": 100.0, "satisfied": False}
    out, _ = call(calc, "compare_to_threshold",
                  {"patient_value": 1, "comparator": "between", "threshold": 0, "threshold_upper": 1, "upper_limit_of_normal": None})
    assert out["satisfied"] is True
    out, is_error = call(calc, "compare_to_threshold", {"patient_value": 1, "comparator": "between", "threshold": 0, **base})
    assert is_error


def test_unit_conversion_known_and_unknown():
    calc = calculator_tools()
    out, _ = call(calc, "convert_lab_unit", {"analyte": "creatinine", "value": 88.42, "from_unit": "µmol/L", "to_unit": "mg/dL"})
    assert out == {"value": 1.0, "unit": "mg/dL"}
    out, _ = call(calc, "convert_lab_unit", {"analyte": "blood_cell_count", "value": 1500, "from_unit": "cells/mm3", "to_unit": "x 10^9/L"})
    assert out["value"] == 1.5
    out, _ = call(calc, "convert_lab_unit", {"analyte": "hemoglobin", "value": 118, "from_unit": "g/L", "to_unit": "g/dL"})
    assert out["value"] == 11.8
    out, is_error = call(calc, "convert_lab_unit", {"analyte": "creatinine", "value": 1, "from_unit": "mg/dL", "to_unit": "furlongs"})
    assert is_error and "Do not convert by hand" in out


# ---- note tools -----------------------------------------------------------


def test_note_search_and_read_return_exact_segment_text():
    note = "Age 62.\nShe denies interstitial lung disease.\nLabs: ALT 28 U/L.\nPlan: discuss trial."
    segments = segment_note(note)
    nt = note_tools(segments)
    out, _ = call(nt, "search_note", {"query": "ILD interstitial pneumonitis"})
    assert [m["segment_id"] for m in out["matches"]] == ["S2"]
    assert out["matches"][0]["text"] == "She denies interstitial lung disease."
    out, _ = call(nt, "search_note", {"query": "hepatitis"})
    assert out["matches"] == [] and out["segments_in_note"] == 4
    out, _ = call(nt, "read_segments", {"first_segment": 2, "last_segment": 99})
    assert [s["segment_id"] for s in out["segments"]] == ["S2", "S3", "S4"]
    _, is_error = call(nt, "read_segments", {"first_segment": 9, "last_segment": 12})
    assert is_error


# ---- pipeline wiring ------------------------------------------------------

GOOD = {
    "INC-01": judgment("INC-01", "met", ["62-year-old woman"]),
    "INC-02": judgment("INC-02", "met", ["62-year-old"]),
    "INC-03": judgment("INC-03", "met", ["ECOG performance status 1"]),
    "EXC-01": judgment("EXC-01", "not_met", ["She denies any history of interstitial lung disease"], segment="S3"),
    "EXC-02": judgment("EXC-02", "unknown"),
}
THRESHOLD_CALL = ("compare_to_threshold", {"patient_value": 62, "comparator": ">=", "threshold": 18,
                                            "threshold_upper": None, "upper_limit_of_normal": None})


def test_short_note_gets_calculators_only_and_calls_are_recorded(settings, patient, trial):
    llm = FakeLLM(GOOD, tool_script=[THRESHOLD_CALL])
    run = run_match(patient, trial, settings, llm)
    judging = [c for c in llm.calls if c["schema"] is LLMJudgmentBatch]
    assert all("search_note" not in c["tools"] and "compare_to_threshold" in c["tools"] for c in judging)
    assert all("calculator tools" in c["system"] and "search_note" not in c["system"] for c in judging)
    assert len(run.tool_calls) == 1
    recorded = run.tool_calls[0]
    assert recorded.tool == "compare_to_threshold" and json.loads(recorded.output)["satisfied"] is True
    assert recorded.criterion_ids == ["INC-01", "INC-02", "INC-03", "EXC-01", "EXC-02"]


def test_long_note_also_gets_search_tools(settings, patient, trial):
    import dataclasses

    llm = FakeLLM(GOOD)
    run_match(patient, trial, dataclasses.replace(settings, full_note_max_chars=10), llm)
    judging = [c for c in llm.calls if c["schema"] is LLMJudgmentBatch][0]
    assert {"search_note", "read_segments", "check_time_window"} <= set(judging["tools"])
    assert "use search_note and read_segments" in judging["user_blocks"][0]
    assert "Before answering \"unknown\"" in judging["system"]


def test_tools_can_be_switched_off(settings, patient, trial):
    import dataclasses

    llm = FakeLLM(GOOD, tool_script=[THRESHOLD_CALL])
    run = run_match(patient, trial, dataclasses.replace(settings, judge_tools=False), llm)
    judging = [c for c in llm.calls if c["schema"] is LLMJudgmentBatch][0]
    assert judging["tools"] == [] and "Tools\n" not in judging["system"] and run.tool_calls == []


# ---- model wrapper: the tool loop against a scripted client ----------------


def _message(stop_reason, *blocks):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), usage=usage, parsed_output=None)


def _tool_use(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


class _Stream:
    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        return SimpleNamespace(get_final_message=lambda: self.reply)

    def __exit__(self, *exc):
        return False


class ScriptedClient:
    """Stands in for anthropic.Anthropic: returns queued replies (the last one repeats) and records each request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def stream(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return _Stream(self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


def scripted(replies):
    return ScriptedClient(replies)


FINAL_JSON = json.dumps({"judgments": [judgment("INC-01", "unknown")]})


def test_wrapper_runs_tools_and_returns_all_results_in_one_message(settings):
    thinking = SimpleNamespace(type="thinking", model_dump=lambda **k: {"type": "thinking", "thinking": "", "signature": "sig"})
    client = scripted([
        _message(
            "tool_use",
            thinking,
            _tool_use("t1", "date_difference", {"earlier_date": "2026-08-04", "later_date": "2026-09-18"}),
            _tool_use("t2", "date_difference", {"earlier_date": "bad", "later_date": "2026-09-18"}),
            _tool_use("t3", "no_such_tool", {}),
        ),
        _message("end_turn", SimpleNamespace(type="text", text=FINAL_JSON)),
    ])
    llm = AnthropicLLM(settings, client=client)
    trace = []
    result = llm.structured(system="s", user_blocks=["note", "criteria"], schema=LLMJudgmentBatch,
                            tools=calculator_tools(), trace=trace)

    assert result.judgments[0].criterion_id == "INC-01"
    first, second = client.requests
    assert [t["name"] for t in first["tools"]][:2] == ["date_difference", "check_time_window"]
    assert "tool_choice" not in first  # forced tool use is rejected by the default model
    assistant, results = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert [b["tool_use_id"] for b in results["content"]] == ["t1", "t2", "t3"]
    assert [b["is_error"] for b in results["content"]] == [False, True, True]
    assert json.loads(results["content"][0]["content"])["days"] == 45
    assert [c["tool"] for c in trace] == ["date_difference", "date_difference", "no_such_tool"]
    assert llm.usage_summary()["requests"] == 2


def test_wrapper_sends_no_tools_field_when_none_are_offered(settings):
    client = scripted([_message("end_turn", SimpleNamespace(type="text", text=FINAL_JSON))])
    AnthropicLLM(settings, client=client).structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)
    assert "tools" not in client.requests[0]


def test_wrapper_stops_a_model_that_never_stops_calling_tools(settings):
    looping = _message("tool_use", _tool_use("t", "date_difference", {"earlier_date": "2026-01-01", "later_date": "2026-01-02"}))
    client = scripted([looping])
    with pytest.raises(LLMError, match="still calling tools"):
        AnthropicLLM(settings, client=client).structured(
            system="s", user_blocks=["x"], schema=LLMJudgmentBatch, tools=calculator_tools()
        )
    assert len(client.requests) == MAX_TOOL_TURNS


# ---- trial tools and the finder --------------------------------------------

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT04035486", "briefTitle": "FLAURA2"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "conditionsModule": {"conditions": ["NSCLC"]},
        "designModule": {"phases": ["PHASE3"]},
        "armsInterventionsModule": {"interventions": [{"name": "Osimertinib"}]},
        "eligibilityModule": {"sex": "ALL", "minimumAge": "18 Years"},
    }
}


def test_search_trials_builds_the_query_and_records_what_it_returned(settings, monkeypatch):
    captured = {}

    def fake_get(url, params, **kwargs):
        captured.update(params)
        return httpx.Response(200, json={"totalCount": 31, "studies": [STUDY]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(tools_module.httpx, "get", fake_get)
    seen = {}
    out, is_error = call(trial_tools(settings, seen), "search_trials", {
        "condition": "non-small cell lung cancer", "other_terms": "EGFR first-line", "country": None,
        "recruiting_only": True, "max_results": 500,
    })
    assert not is_error and out["total_matching"] == 31
    assert out["trials"][0] == {
        "nct_id": "NCT04035486", "title": "FLAURA2", "status": "RECRUITING", "conditions": ["NSCLC"],
        "phases": ["PHASE3"], "interventions": ["Osimertinib"], "minimum_age": "18 Years", "maximum_age": None, "sex": "ALL",
    }
    assert captured["query.cond"] == "non-small cell lung cancer" and captured["query.term"] == "EGFR first-line"
    assert captured["filter.overallStatus"] == "RECRUITING" and captured["pageSize"] == 20
    assert "query.locn" not in captured
    assert list(seen) == ["NCT04035486"]


def test_search_failure_is_reported_to_the_model_not_raised(settings, monkeypatch):
    def fake_get(url, params, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(tools_module.httpx, "get", fake_get)
    out, is_error = call(trial_tools(settings, {}), "search_trials", {
        "condition": "x", "other_terms": None, "country": None, "recruiting_only": False, "max_results": 5,
    })
    assert is_error and "ConnectError" in out


class FinderLLM:
    """Pretends to search once, then names one trial it found and one it did not."""

    def structured(self, *, system, user_blocks, schema, tools=None, trace=None):
        assert schema is LLMShortlist
        output, is_error = {t.name: t for t in tools}["search_trials"].run({
            "condition": "lung adenocarcinoma", "other_terms": "EGFR", "country": None,
            "recruiting_only": True, "max_results": 10,
        })
        trace.append({"tool": "search_trials", "input": {}, "output": output, "is_error": is_error})
        return LLMShortlist(
            candidates=[
                {"nct_id": "nct04035486", "why_relevant": "EGFR-mutated, untreated.", "concerns": " "},
                {"nct_id": "NCT99999999", "why_relevant": "Remembered, never searched.", "concerns": None},
                {"nct_id": "NCT04035486", "why_relevant": "Duplicate.", "concerns": None},
            ],
            search_notes="One search.",
        )


def test_finder_keeps_only_trials_a_tool_returned(settings, monkeypatch):
    monkeypatch.setattr(
        tools_module.httpx, "get",
        lambda url, params, **kw: httpx.Response(200, json={"totalCount": 1, "studies": [STUDY]}, request=httpx.Request("GET", url)),
    )
    patient = Patient(id="sample-test", source="synthetic", text="62-year-old woman with EGFR-mutated lung adenocarcinoma.")
    result = find_trials(patient, settings, FinderLLM(), max_candidates=5)
    assert [c.nct_id for c in result.candidates] == ["NCT04035486"]
    assert result.candidates[0].title == "FLAURA2" and result.candidates[0].concerns is None
    assert result.rejected_ids == ["NCT99999999"]
    assert [c["tool"] for c in result.tool_calls] == ["search_trials"]


def test_unknown_trial_id_is_reported_as_not_found(settings, monkeypatch):
    def fake_get(url, **kwargs):
        return httpx.Response(400, text="Parameter `nctId` has incorrect format", request=httpx.Request("GET", url))

    monkeypatch.setattr("trialmatch.ctgov.httpx.get", fake_get)
    out, is_error = call(trial_tools(settings, {}), "get_trial_criteria", {"nct_id": "NCT00000000"})
    assert is_error and out == "NCT00000000 was not found on ClinicalTrials.gov"
    out, is_error = call(trial_tools(settings, {}), "get_trial_criteria", {"nct_id": "FLAURA2"})
    assert is_error and "Not a valid NCT number" in out
