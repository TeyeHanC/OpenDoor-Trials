"""Lexical (BM25) evidence retrieval over note segments.

BM25 ranks segments for every criterion. Short notes are still passed to the
model in full (see ``select_segments``): lexical search misses paraphrases
("renal impairment" vs "creatinine 2.4"), and a retrieval miss would otherwise
turn into a wrong "insufficient information" verdict.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import AtomicCriterion, RetrievedSegment, Segment

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in is it its of on or that the this to was "
    "were with within without no not any other than who whom must should may patient patients "
    "subject subjects participant participants history prior".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, segments: list[Segment], k1: float = 1.5, b: float = 0.75):
        self.segments = segments
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(s.text)) for s in segments]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if segments else 0.0
        doc_freq: Counter[str] = Counter()
        for counts in self.term_counts:
            doc_freq.update(counts.keys())
        n = len(segments)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def rank(self, query: str) -> list[RetrievedSegment]:
        """All segments, best first. Segments sharing no term with the query score 0."""
        terms = set(tokenize(query))
        scored = []
        for segment, counts, length in zip(self.segments, self.term_counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scored.append(RetrievedSegment(segment_id=segment.id, score=round(score, 4)))
        return sorted(scored, key=lambda r: -r.score)


def criterion_query(criterion: AtomicCriterion) -> str:
    """Criterion text plus both spellings of each concept (original and normalized)."""
    parts = [criterion.text]
    for concept in criterion.concepts:
        parts.append(concept.original)
        if concept.normalized:
            parts.append(concept.normalized)
    if criterion.threshold:
        parts.append(criterion.threshold.parameter)
    return " ".join(parts)


def select_segments(
    ranked: list[RetrievedSegment],
    segments: list[Segment],
    *,
    note_length: int,
    full_note_max_chars: int,
    top_k: int,
) -> list[RetrievedSegment]:
    """Which ranked segments the model gets to see for one criterion.

    Short note: everything. Long note: the top-k matching segments, plus the
    first segment, which usually carries age and sex in words BM25 cannot match
    to a criterion such as "Age >= 18 years".
    """
    if note_length <= full_note_max_chars:
        return ranked
    chosen = [r for r in ranked[:top_k] if r.score > 0]
    if segments and all(r.segment_id != segments[0].id for r in chosen):
        first = next(r for r in ranked if r.segment_id == segments[0].id)
        chosen.append(first)
    return chosen
