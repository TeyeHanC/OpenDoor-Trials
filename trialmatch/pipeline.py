"""End-to-end match of one patient note against one trial."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from .config import Settings
from .decompose import decompose_trial
from .judge import judge_batch
from .llm import StructuredLLM
from .models import AtomicCriterion, CitationStats, CriterionResult, Patient, RetrievedSegment, RunResult, Trial
from .retrieval import BM25Index, criterion_query, select_segments
from .segment import segment_note
from .summarize import summarize
from .verify import unresolved_result, verify_judgment

Progress = Callable[[str], None]


def make_run_id(patient_id: str, nct_id: str) -> str:
    return f"{patient_id}__{nct_id}"


def citation_stats(results: list[CriterionResult]) -> CitationStats:
    verified = sum(len(r.evidence) for r in results)
    rejected = sum(len(r.rejected_quotes) for r in results)
    return CitationStats(
        quotes_total=verified + rejected,
        quotes_verified=verified,
        judgments_needing_evidence=sum(1 for r in results if r.model_verdict != "unknown"),
        judgments_downgraded=sum(1 for r in results if r.downgraded),
    )


def run_match(
    patient: Patient,
    trial: Trial,
    settings: Settings,
    llm: StructuredLLM,
    progress: Progress = lambda message: None,
) -> RunResult:
    progress(f"Decomposing eligibility criteria for {trial.nct_id}")
    criteria = decompose_trial(trial, settings, llm)

    segments = segment_note(patient.text, settings.segment_max_chars)
    by_id = {s.id: s for s in segments}
    complete = len(patient.text) <= settings.full_note_max_chars
    index = BM25Index(segments)
    retrieved: dict[str, list[RetrievedSegment]] = {
        c.id: select_segments(
            index.rank(criterion_query(c)),
            segments,
            note_length=len(patient.text),
            full_note_max_chars=settings.full_note_max_chars,
            top_k=settings.top_k_segments,
        )
        for c in criteria
    }

    size = settings.judge_batch_size
    batches = [criteria[i : i + size] for i in range(0, len(criteria), size)]

    def judge(batch: list[AtomicCriterion]) -> list[CriterionResult]:
        # A batch sees the union of its criteria's segments, in note order.
        wanted = {r.segment_id for c in batch for r in retrieved[c.id]}
        batch_segments = [s for s in segments if s.id in wanted]
        judgments = judge_batch(patient, batch_segments, batch, llm, complete=complete)
        results = []
        for criterion in batch:
            # Only segments that matched the query are recorded as "retrieved".
            hits = [r for r in retrieved[criterion.id] if r.score > 0 and r.segment_id in by_id]
            if judgments is None:
                results.append(
                    unresolved_result(criterion, hits, flag="model_declined", reason="The model declined to assess this criterion.")
                )
            elif criterion.id not in judgments:
                results.append(
                    unresolved_result(criterion, hits, flag="no_judgment_returned", reason="The model returned no judgment for this criterion.")
                )
            else:
                results.append(verify_judgment(judgments[criterion.id], criterion, patient.text, segments, hits))
        return results

    progress(f"Judging {len(criteria)} criteria in {len(batches)} batches")
    with ThreadPoolExecutor(max_workers=settings.judge_workers) as pool:
        results = [r for batch_results in pool.map(judge, batches) for r in batch_results]

    return RunResult(
        run_id=make_run_id(patient.id, trial.nct_id),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=settings.model,
        patient=patient,
        segments=segments,
        trial=trial,
        criteria=results,
        summary=summarize(results),
        citation_stats=citation_stats(results),
    )
