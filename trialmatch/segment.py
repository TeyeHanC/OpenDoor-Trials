"""Split a note into segments while keeping exact character offsets.

Segments are paragraphs; a paragraph longer than ``max_chars`` is further cut
at sentence boundaries (TREC patient descriptions are one long paragraph).
Every segment satisfies ``note[seg.start:seg.end] == seg.text`` so evidence
can later be highlighted in the untouched original text.
"""

from __future__ import annotations

import re

from .models import Segment

_LINE_RE = re.compile(r"[^\n]+")
# A sentence ends at . ! or ? followed by whitespace and an upper-case letter or
# digit. Abbreviations ("Dr. Smith") produce an extra cut, which is harmless.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _sentence_spans(text: str, offset: int) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        spans.append((offset + start, offset + match.start()))
        start = match.end()
    spans.append((offset + start, offset + len(text)))
    return spans


def _pack(spans: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Greedily merge consecutive sentence spans up to ``max_chars``."""
    packed: list[tuple[int, int]] = []
    for start, end in spans:
        if packed and end - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed


def segment_note(note: str, max_chars: int = 400) -> list[Segment]:
    spans: list[tuple[int, int]] = []
    for line in _LINE_RE.finditer(note):
        raw = line.group()
        stripped = raw.strip()
        if not stripped:
            continue
        start = line.start() + (len(raw) - len(raw.lstrip()))
        if len(stripped) <= max_chars:
            spans.append((start, start + len(stripped)))
        else:
            spans.extend(_pack(_sentence_spans(stripped, start), max_chars))
    return [
        Segment(id=f"S{i}", start=start, end=end, text=note[start:end])
        for i, (start, end) in enumerate(spans, start=1)
    ]
