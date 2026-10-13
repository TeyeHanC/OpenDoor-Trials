from trialmatch.ctgov import normalize_nct_id, parse_study
from trialmatch.models import AtomicCriterion
from trialmatch.retrieval import BM25Index, criterion_query, select_segments
from trialmatch.segment import segment_note

NOTE = (
    "Header line\n\n"
    "  A 62-year-old woman with stage IV lung adenocarcinoma. She quit smoking in 2015. "
    "Dr. Lee reviewed the CT. ECOG performance status 1.\n"
    "Labs: creatinine 0.9 mg/dL.\n"
)


def test_segments_preserve_offsets():
    for max_chars in (40, 400):
        segments = segment_note(NOTE, max_chars=max_chars)
        assert segments
        for seg in segments:
            assert NOTE[seg.start : seg.end] == seg.text
            assert seg.text == seg.text.strip() and seg.text
        assert [s.start for s in segments] == sorted(s.start for s in segments)


def test_long_paragraph_is_split_at_sentences():
    segments = segment_note(NOTE, max_chars=60)
    texts = [s.text for s in segments]
    assert "Header line" in texts
    assert any(t.startswith("ECOG performance status") for t in texts)
    assert len(segments) > 3


def test_bm25_ranks_matching_segment_first():
    segments = segment_note(NOTE, max_chars=60)
    index = BM25Index(segments)
    top = index.rank("serum creatinine below 1.5 mg/dL")[0]
    assert "creatinine" in next(s.text for s in segments if s.id == top.segment_id)


def test_select_segments_returns_top_k_matches_plus_first_segment():
    segments = segment_note(NOTE, max_chars=60)
    ranked = BM25Index(segments).rank("creatinine")
    limited = select_segments(ranked, segments, top_k=2)
    ids = [r.segment_id for r in limited]
    assert segments[0].id in ids and len(ids) == 2


def test_criterion_query_includes_normalized_concepts():
    criterion = AtomicCriterion(
        id="EXC-01",
        type="exclusion",
        text="Currently taking Coumadin",
        source_text="Coumadin",
        concepts=[{"original": "Coumadin", "normalized": "warfarin", "category": "drug"}],
    )
    assert "warfarin" in criterion_query(criterion)


def test_parse_study_and_nct_validation():
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000102", "briefTitle": "T"},
            "conditionsModule": {"conditions": ["CAH"]},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n\n* x", "minimumAge": "14 Years", "sex": "ALL"},
        }
    }
    trial = parse_study(payload, "2026-01-01T00:00:00+00:00")
    assert trial.nct_id == "NCT00000102" and trial.minimum_age == "14 Years" and trial.maximum_age is None
    assert normalize_nct_id(" nct00000102 ") == "NCT00000102"
    try:
        normalize_nct_id("NCT123")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
