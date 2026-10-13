"""End-to-end match of one patient note against one trial."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from .config import Settings
from .decompose import decompose_trial
from .judge import judge_batch
from .llm import StructuredLLM
from .models import (
    AtomicCriterion,
    CitationStats,
    CriterionResult,
    Patient,
    RetrievedSegment,
    RunResult,
    ToolCall,
    Trial,
)
from .retrieval import BM25Index, criterion_query, select_segments
from .segment import segment_note
from .summarize import summarize
from .tools import note_tools
from .verify import unresolved_result, verify_judgment

Progress = Callable[[str], None]


def make_run_id(patient_id: str, nct_id: str, suffix: str = "") -> str:
    return f"{patient_id}__{nct_id}{suffix}"


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
    run_suffix: str = "",
) -> RunResult:
    progress(f"Decomposing eligibility criteria for {trial.nct_id}")
    criteria = decompose_trial(trial, settings, llm)

    segments = segment_note(patient.text, settings.segment_max_chars)
    by_id = {s.id: s for s in segments}
    # Short notes go to the model whole; only long ones are narrowed by retrieval.
    complete = len(patient.text) <= settings.full_note_max_chars
    retrieved: dict[str, list[RetrievedSegment]] = {c.id: [] for c in criteria}
    if not complete:
        index = BM25Index(segments)
        for c in criteria:
            retrieved[c.id] = select_segments(index.rank(criterion_query(c)), segments, settings.top_k_segments)

    # Search tools only when the model cannot see the whole note. Date and threshold arithmetic is
    # not a tool: the model reports its inputs and the program recomputes them (checks.py).
    tools = note_tools(segments) if settings.judge_tools and not complete else []

    size = settings.judge_batch_size
    batches = [criteria[i : i + size] for i in range(0, len(criteria), size)]

    def judge(batch: list[AtomicCriterion]) -> tuple[list[CriterionResult], list[ToolCall]]:
        if complete:
            batch_segments = segments
        else:
            # A batch sees the union of its criteria's segments, in note order.
            wanted = {r.segment_id for c in batch for r in retrieved[c.id]}
            batch_segments = [s for s in segments if s.id in wanted]
        trace: list[dict] = []
        judgments = judge_batch(patient, batch_segments, batch, llm, complete=complete, tools=tools, trace=trace)
        calls = [ToolCall(criterion_ids=[c.id for c in batch], **call) for call in trace]
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
        return results, calls

    progress(f"Judging {len(criteria)} criteria in {len(batches)} batches")
    with ThreadPoolExecutor(max_workers=settings.judge_workers) as pool:
        judged = list(pool.map(judge, batches))
    results = [r for batch_results, _ in judged for r in batch_results]
    tool_calls = [call for _, batch_calls in judged for call in batch_calls]

    return RunResult(
        run_id=make_run_id(patient.id, trial.nct_id, run_suffix),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=settings.model,
        note_coverage="full" if complete else "retrieved",
        judge_batch_size=settings.judge_batch_size,
        patient=patient,
        segments=segments,
        trial=trial,
        criteria=results,
        summary=summarize(results),
        citation_stats=citation_stats(results),
        tool_calls=tool_calls,
    )
