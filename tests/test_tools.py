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


def test_short_note_is_judged_without_any_tools(settings, patient, trial):
    llm = FakeLLM(GOOD)
    run = run_match(patient, trial, settings, llm)
    judging = [c for c in llm.calls if c["schema"] is LLMJudgmentBatch]
    assert all(c["tools"] == [] and "search_note" not in c["system"] for c in judging)
    assert "Arithmetic the program re-checks" in judging[0]["system"] and run.tool_calls == []


def test_long_note_gets_search_tools_and_their_calls_are_recorded(settings, patient, trial):
    import dataclasses

    llm = FakeLLM(GOOD, tool_script=[("search_note", {"query": "interstitial lung disease ILD"})])
    run = run_match(patient, trial, dataclasses.replace(settings, full_note_max_chars=10), llm)
    judging = [c for c in llm.calls if c["schema"] is LLMJudgmentBatch][0]
    assert judging["tools"] == ["search_note", "read_segments"]
    assert "use search_note and read_segments" in judging["user_blocks"][0]
    assert "Before answering \"unknown\"" in judging["system"]
    assert run.tool_calls[0].tool == "search_note" and "denies" in run.tool_calls[0].output
    assert run.tool_calls[0].criterion_ids == ["INC-01", "INC-02", "INC-03", "EXC-01", "EXC-02"]


def test_note_tools_can_be_switched_off(settings, patient, trial):
    import dataclasses

    llm = FakeLLM(GOOD)
    run_match(patient, trial, dataclasses.replace(settings, full_note_max_chars=10, judge_tools=False), llm)
    assert [c for c in llm.calls if c["schema"] is LLMJudgmentBatch][0]["tools"] == []


# ---- arithmetic the program recomputes --------------------------------------

AGE_OK = {"value": 62, "value_unit": "years", "comparator": ">=", "threshold": 18, "threshold_upper": None,
          "threshold_unit": "years", "reference_limit": None, "reference_limit_assumed": False, "analyte": None, "satisfied": True}


def _one(settings, patient, trial, criterion_id, scripted_judgment):
    run = run_match(patient, trial, settings, FakeLLM(dict(GOOD, **{criterion_id: scripted_judgment})))
    return {c.criterion_id: c for c in run.criteria}[criterion_id], run


def test_agreeing_threshold_check_is_recorded_as_confirmed(settings, patient, trial):
    c, _ = _one(settings, patient, trial, "INC-02", judgment("INC-02", "met", ["62-year-old"], threshold_check=AGE_OK))
    assert c.verdict == "met" and c.flags == [] and c.confidence == "high"
    assert c.checks[0]["status"] == "confirmed" and c.checks[0]["computed"] is True
    assert c.checks[0]["detail"] == "62 >= 18.0 years is true"


def test_unit_spellings_that_mean_the_same_are_not_treated_as_a_conversion(settings, patient, trial):
    spelled = dict(AGE_OK, value_unit="year-old", threshold_unit="Years")
    c, _ = _one(settings, patient, trial, "INC-02", judgment("INC-02", "met", ["62-year-old"], threshold_check=spelled))
    assert c.checks[0]["status"] == "confirmed" and c.flags == [] and c.confidence == "high"

    from trialmatch.checks import _same_unit

    assert _same_unit("U/L", "u/l") and _same_unit("yrs", "years") and _same_unit("x 10^9/L", "10^9/L")
    assert not _same_unit("g/dL", "g/L") and not _same_unit("mg/dL", "umol/L")


def test_wrong_arithmetic_sets_the_determination_aside(settings, patient, trial):
    # The model claims ALT 28 U/L exceeds 2.5 x ULN (40): 28 > 100 is false.
    wrong = {"value": 28, "value_unit": "U/L", "comparator": ">", "threshold": 2.5, "threshold_upper": None,
             "threshold_unit": None, "reference_limit": 40, "reference_limit_assumed": False, "analyte": None, "satisfied": True}
    c, run = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "met", ["ALT 28 U/L"], segment="S5", threshold_check=wrong))
    assert c.model_verdict == "met" and c.verdict == "unknown" and c.effect == "unresolved" and c.confidence == "low"
    assert c.flags == ["calculation_mismatch"] and c.checks[0]["status"] == "mismatch"
    assert "28 > 100.0 (2.5 x 40)" in c.checks[0]["detail"] and "did not match the program's" in c.missing_info
    assert run.summary.overall == "needs_more_info"  # a wrong "exclusion applies" must not block enrollment


def test_time_window_is_recomputed_from_the_dates(settings, patient, trial):
    claims_recent = {"event_date": "2026-08-04", "reference_date": "2026-09-18", "window_amount": 28, "window_unit": "days", "within_window": True}
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "met", ["ALT 28 U/L"], segment="S5", time_check=claims_recent))
    assert c.verdict == "unknown" and c.checks[0]["computed"] is False
    assert "45 days before 2026-09-18" in c.checks[0]["detail"] and "outside" in c.checks[0]["detail"]
    # 2026-08-04 is not written in the fixture note, so the input is also flagged on a correct answer.
    correct = dict(claims_recent, within_window=False)
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "not_met", ["ALT 28 U/L"], segment="S5", time_check=correct))
    assert c.verdict == "not_met" and c.flags == ["calculation_input_not_in_note"] and c.confidence == "low"
    assert c.checks[0]["status"] == "confirmed" and c.checks[0]["inputs_not_in_note"] == ["2026-08-04"]


def test_units_are_converted_by_the_program_or_the_check_is_marked_uncheckable(settings, patient, trial):
    from trialmatch.checks import run_threshold_check
    from trialmatch.models import LLMThresholdCheck

    base = {"comparator": "<", "threshold": 90, "threshold_upper": None, "threshold_unit": "g/L", "reference_limit": None,
            "reference_limit_assumed": False}
    hb = LLMThresholdCheck(value=11.8, value_unit="g/dL", analyte="hemoglobin", satisfied=False, **base)
    done = run_threshold_check(hb, "Hemoglobin 11.8 g/dL")
    assert done.status == "confirmed" and done.detail == "11.8 g/dL = 118 g/L; 118 < 90.0 g/L is false" and done.inputs_not_in_note == []
    no_analyte = run_threshold_check(hb.model_copy(update={"analyte": None}), "Hemoglobin 11.8 g/dL")
    assert no_analyte.status == "not_checkable" and no_analyte.computed is None

    partial = {"event_date": "2014", "reference_date": "2026-09-18", "window_amount": 4, "window_unit": "weeks", "within_window": False}
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "not_met", ["ALT 28 U/L"], segment="S5", time_check=partial))
    assert c.verdict == "not_met" and c.flags == ["calculation_not_checkable"] and c.confidence == "low"


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


# ---- OpenAI-compatible endpoint ---------------------------------------------


def _openai_llm(settings, handler, monkeypatch, model="gpt-test"):
    import dataclasses

    from trialmatch.llm_openai import OpenAICompatLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.test/v1/")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("trialmatch.llm_openai.time.sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatLLM(dataclasses.replace(settings, model=model), client=client)


def _chat(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 3}})


def test_provider_is_chosen_from_the_model_name(settings):
    import dataclasses

    from trialmatch.llm import resolve_provider

    assert resolve_provider(dataclasses.replace(settings, model="claude-opus-5-5")) == "anthropic"
    assert resolve_provider(dataclasses.replace(settings, model="gpt-6-luna")) == "openai"
    assert resolve_provider(dataclasses.replace(settings, model="gpt-6-luna", provider="anthropic")) == "anthropic"
    with pytest.raises(ValueError):
        resolve_provider(dataclasses.replace(settings, provider="azure"))


def test_openai_endpoint_tool_loop_and_strict_schema(settings, monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((str(request.url), request.headers.get("authorization"), body))
        if len(requests) == 1:
            return _chat(tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "date_difference", "arguments": json.dumps({"earlier_date": "2026-08-04", "later_date": "2026-09-18"})}}],
                finish="tool_calls")
        return _chat(content=FINAL_JSON)

    llm = _openai_llm(settings, handler, monkeypatch)
    trace = []
    result = llm.structured(system="s", user_blocks=["note", "criteria"], schema=LLMJudgmentBatch,
                            tools=calculator_tools(), trace=trace)
    assert result.judgments[0].criterion_id == "INC-01"
    url, auth, first = requests[0]
    assert url == "https://gateway.test/v1/chat/completions" and auth == "Bearer test-key"
    assert first["model"] == "gpt-test" and first["response_format"]["json_schema"]["strict"] is True
    assert first["tools"][0]["function"]["strict"] is True and first["messages"][1]["content"] == "note\n\ncriteria"
    second = requests[1][2]["messages"]
    assert second[2]["tool_calls"][0]["id"] == "c1"
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "c1" and json.loads(second[3]["content"])["days"] == 45
    assert trace[0]["tool"] == "date_difference" and llm.usage_summary() == {"requests": 2, "input_tokens": 14, "output_tokens": 6}


def test_openai_endpoint_falls_back_when_strict_schema_is_rejected(settings, monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        kind = (body.get("response_format") or {}).get("type")
        seen.append(kind)
        if kind in ("json_schema", "json_object"):
            return httpx.Response(400, json={"error": {"message": "response_format not supported"}})
        assert "JSON Schema" in body["messages"][0]["content"]
        assert all("strict" not in tool["function"] for tool in body.get("tools", []))
        return _chat(content="Here you go:\n```json\n" + FINAL_JSON + "\n```")

    llm = _openai_llm(settings, handler, monkeypatch)
    result = llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch, tools=calculator_tools())
    assert result.judgments[0].verdict == "unknown"
    # The strict request is tried twice: once more without the usage option before the schema is weakened.
    assert seen == ["json_schema", "json_schema", "json_object", None] and llm.mode == "prompt"
    llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)
    assert seen[4:] == [None]  # the accepted mode is remembered


def test_openai_endpoint_repairs_an_invalid_reply_once(settings, monkeypatch):
    replies = [_chat(content='{"judgments": [{"criterion_id": "INC-01"}]}'), _chat(content=FINAL_JSON)]
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return replies[len(bodies) - 1]

    llm = _openai_llm(settings, handler, monkeypatch)
    assert llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch).judgments[0].criterion_id == "INC-01"
    assert "did not match the required JSON schema" in bodies[1]["messages"][-1]["content"]

    always_bad = _openai_llm(settings, lambda request: _chat(content="not json"), monkeypatch)
    with pytest.raises(LLMError, match="after one retry"):
        always_bad.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)


def test_openai_endpoint_retries_transient_errors_and_reports_auth_failure(settings, monkeypatch):
    statuses = iter([429, 503, 200])

    def flaky(request):
        status = next(statuses)
        return _chat(content=FINAL_JSON) if status == 200 else httpx.Response(status, text="busy")

    assert _openai_llm(settings, flaky, monkeypatch).structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)
    denied = _openai_llm(settings, lambda request: httpx.Response(401, text="bad key"), monkeypatch)
    with pytest.raises(LLMError, match="rejected the API key"):
        denied.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)


def test_openai_endpoint_reads_a_streamed_answer_with_tool_calls(settings, monkeypatch):
    def sse(*chunks):
        text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return sse(
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "date_difference", "arguments": '{"earlier_date": "2026-08-04",'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "later_date": "2026-09-18"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}},
            )
        half = len(FINAL_JSON) // 2
        return sse(
            {"choices": [{"delta": {"content": FINAL_JSON[:half]}}]},
            {"choices": [{"delta": {"content": FINAL_JSON[half:]}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 9}},
        )

    llm = _openai_llm(settings, handler, monkeypatch)
    trace = []
    result = llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch, tools=calculator_tools(), trace=trace)
    assert result.judgments[0].criterion_id == "INC-01"
    assert bodies[0]["stream"] is True and bodies[0]["stream_options"] == {"include_usage": True}
    assert json.loads(trace[0]["output"])["days"] == 45 and bodies[1]["messages"][2]["tool_calls"][0]["id"] == "c1"
    assert llm.usage_summary() == {"requests": 2, "input_tokens": 31, "output_tokens": 13}


def test_openai_endpoint_drops_the_usage_option_before_weakening_the_schema(settings, monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return _chat(content=FINAL_JSON)

    llm = _openai_llm(settings, handler, monkeypatch)
    llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)
    assert llm.mode == "schema"


def test_gateway_timeout_page_is_reported_briefly(settings, monkeypatch):
    llm = _openai_llm(settings, lambda request: httpx.Response(524, text="<!DOCTYPE html><html>...</html>"), monkeypatch)
    with pytest.raises(LLMError, match="524: the gateway returned an HTML error page \\(response timeout\\)"):
        llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)


def test_openai_endpoint_turns_reasoning_off_only_for_tool_requests_when_told_to(settings, monkeypatch):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("tools") and body.get("reasoning_effort") != "none":
            return httpx.Response(400, json={"error": {"message": "model requires Responses for tool calls with reasoning; use reasoning_effort=none"}})
        return _chat(content=FINAL_JSON)

    llm = _openai_llm(settings, handler, monkeypatch)
    llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch, tools=calculator_tools())
    llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)
    assert [b.get("reasoning_effort") for b in bodies] == [None, "none", None]
    # The strict schema and the usage option survive: only reasoning was the problem.
    assert llm.mode == "schema" and all("stream_options" in b for b in bodies)


def test_openai_endpoint_resends_a_request_whose_stream_was_cut(settings, monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            cut = 'data: {"choices": [{"delta": {"content": "{\\"judg"}}]}\n\ndata: {"error": {"message": "Upstream response stream was interrupted"}}\n\n'
            return httpx.Response(200, text=cut, headers={"content-type": "text/event-stream"})
        return _chat(content=FINAL_JSON)

    llm = _openai_llm(settings, handler, monkeypatch)
    assert llm.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch).judgments[0].criterion_id == "INC-01"
    assert len(attempts) == 3 and llm.usage_summary()["requests"] == 1

    always_cut = _openai_llm(
        settings,
        lambda request: httpx.Response(200, text='data: {"error": {"message": "interrupted"}}\n\n', headers={"content-type": "text/event-stream"}),
        monkeypatch,
    )
    with pytest.raises(LLMError, match="mid-stream"):
        always_cut.structured(system="s", user_blocks=["x"], schema=LLMJudgmentBatch)


# ---- what is left to check: record, screening visit, assumed reference limits ----


def _alt_check(value, satisfied, assumed=True):
    return {"value": value, "value_unit": "U/L", "comparator": ">", "threshold": 2.5, "threshold_upper": None,
            "threshold_unit": None, "reference_limit": 40, "reference_limit_assumed": assumed, "analyte": None, "satisfied": satisfied}


def test_assumed_reference_limit_is_accepted_only_with_a_safe_margin(settings, patient, trial):
    # ALT 28 against 2.5 x an assumed ULN of 40: false at 75, 100 and 125, so the conclusion is safe.
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "not_met", ["ALT 28 U/L"], segment="S5", threshold_check=_alt_check(28, False)))
    assert c.verdict == "not_met" and c.flags == ["assumed_reference_range"] and c.confidence == "medium"
    assert c.checks[0]["status"] == "confirmed" and "typical adult value, not from the note, and the result holds" in c.checks[0]["detail"]

    # A stated limit carries no flag and keeps its confidence.
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "not_met", ["ALT 28 U/L"], segment="S5", threshold_check=_alt_check(28, False, assumed=False)))
    assert c.flags == [] and c.confidence == "high"


def test_conclusion_too_close_to_an_assumed_limit_is_set_aside():
    from trialmatch.checks import run_threshold_check
    from trialmatch.models import LLMThresholdCheck

    # 110 > 100 holds at the typical limit but not if the lab's ULN is 25% higher (125).
    near = run_threshold_check(LLMThresholdCheck(**_alt_check(110, True)), "ALT 110 U/L")
    assert near.computed is True and near.status == "not_robust" and "would change" in near.detail


def test_too_close_to_assumed_limit_becomes_unknown_in_the_run(settings, trial):
    note = "62-year-old woman.\nLabs: ALT 110 U/L."
    patient = Patient(id="sample-near", source="synthetic", text=note)
    scripted = dict(GOOD, **{"EXC-02": judgment("EXC-02", "met", ["ALT 110 U/L"], threshold_check=_alt_check(110, True))})
    for cid in ("INC-03", "EXC-01"):
        scripted[cid] = judgment(cid, "unknown")
    run = run_match(patient, trial, settings, FakeLLM(scripted))
    c = {x.criterion_id: x for x in run.criteria}["EXC-02"]
    assert c.model_verdict == "met" and c.verdict == "unknown" and c.flags == ["assumed_reference_range_too_close"]
    assert "Obtain the laboratory's own reference limit" in c.missing_info and run.summary.counts["blocks"] == 0


def test_screening_visit_criteria_are_listed_apart_and_do_not_hold_back_the_conclusion(settings, patient, trial):
    from trialmatch.models import CriterionResult
    from trialmatch.summarize import summarize

    def result(cid, verdict, source="medical_record", **extra):
        kind = "inclusion" if cid.startswith("INC") else "exclusion"
        effect = "unresolved" if verdict == "unknown" else "supports"
        return CriterionResult(criterion_id=cid, criterion_text=cid, source_text=cid, type=kind, verdict=verdict, effect=effect,
                               evidence=[], rationale="r", confidence="high", model_verdict=verdict, answer_source=source, **extra)

    consent = result("INC-02", "unknown", "screening_visit", missing_info="Ask about contraception.")
    labs = result("EXC-01", "unknown", missing_info="Get hepatitis serology.", established_part="no HIV history")
    only_visit = summarize([result("INC-01", "met"), consent])
    assert only_visit.overall == "possibly_eligible" and only_visit.follow_up == []
    assert [i.criterion_id for i in only_visit.confirm_at_screening] == ["INC-02"]
    assert only_visit.confirm_at_screening[0].text.startswith('Confirm at screening "INC-02": Ask about contraception.')

    both = summarize([result("INC-01", "met"), consent, labs])
    assert both.overall == "needs_more_info"
    assert [i.criterion_id for i in both.follow_up] == ["EXC-01"] and "Already established: no HIV history." in both.follow_up[0].text
    assert both.counts["missing_from_record"] == 1 and both.counts["confirm_at_screening"] == 1 and both.counts["partly_established"] == 1


def test_established_part_and_answer_source_reach_the_stored_result(settings, patient, trial):
    c, _ = _one(settings, patient, trial, "EXC-02", judgment("EXC-02", "unknown", established_part=" no HIV history "))
    assert c.established_part == "no HIV history" and c.answer_source == "medical_record"
    met, _ = _one(settings, patient, trial, "INC-03", judgment("INC-03", "met", ["ECOG performance status 1"], established_part="ignored"))
    assert met.established_part is None
