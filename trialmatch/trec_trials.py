"""Trials as they were when TREC assessors judged them.

TREC Clinical Trials 2021 and 2022 both use one ClinicalTrials.gov snapshot
taken on 2021-04-27, published as five zip archives (about 1.8 GB in total).
Evaluation must read criteria from this snapshot: the live registry record may
have been edited since, and then a "wrong" prediction could simply be a
changed criterion.

Only the trials that are asked for are downloaded. Each archive's table of
contents is read once with HTTP range requests and indexed; after that a trial
costs one small range request and is cached as JSON. If the archives have been
downloaded into ``data/trec/corpus/`` they are read from disk instead.
"""

from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import httpx

from .config import Settings
from .ctgov import TrialNotFound, normalize_nct_id
from .models import Trial

SNAPSHOT_DATE = "2021-04-27"
SOURCE = f"trec_snapshot_{SNAPSHOT_DATE}"
PARTS = (1, 2, 3, 4, 5)
PART_NAME = "ClinicalTrials.2021-04-27.part{part}.zip"
PART_URL = "https://www.trec-cds.org/2021_data/" + PART_NAME


class HttpRangeFile:
    """Read-only, seekable view of a remote file, fetched in ranges on demand."""

    def __init__(self, url: str, client: httpx.Client, block: int = 1 << 16):
        self.url, self.client, self.block = url, client, block
        head = client.head(url, follow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])
        self.pos = 0
        self._buffer_start, self._buffer = 0, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start = self.pos - self._buffer_start
        if not (0 <= start and start + n <= len(self._buffer)):
            end = min(self.size, self.pos + max(n, self.block)) - 1
            response = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, follow_redirects=True)
            if response.status_code != 206:
                raise OSError(f"Server did not honour the range request for {self.url} ({response.status_code})")
            self._buffer_start, self._buffer, start = self.pos, response.content, 0
        self.pos += n
        return self._buffer[start : start + n]


def corpus_dir(settings: Settings) -> Path:
    return settings.trec_dir / "corpus"


def _open_part(settings: Settings, part: int, client: httpx.Client) -> zipfile.ZipFile:
    local = corpus_dir(settings) / PART_NAME.format(part=part)
    if local.is_file():
        return zipfile.ZipFile(local)
    return zipfile.ZipFile(HttpRangeFile(PART_URL.format(part=part), client))


def load_index(settings: Settings, client: httpx.Client) -> dict[str, list]:
    """NCT id -> [part, member name], built once from the archives' tables of contents."""
    path = corpus_dir(settings) / "index.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, list] = {}
    for part in PARTS:
        with _open_part(settings, part, client) as archive:
            for name in archive.namelist():
                stem = name.rsplit("/", 1)[-1]
                if stem.startswith("NCT") and stem.endswith(".xml"):
                    index[stem[:-4]] = [part, name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index), encoding="utf-8")
    return index


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return None if value in ("", "N/A") else value


def parse_snapshot_xml(xml_bytes: bytes) -> Trial:
    root = ET.fromstring(xml_bytes)
    nct_id = root.findtext("id_info/nct_id", "").strip()
    criteria = root.findtext("eligibility/criteria/textblock", "") or ""
    sex = _clean(root.findtext("eligibility/gender"))
    return Trial(
        nct_id=nct_id,
        title=(root.findtext("brief_title") or "").strip(),
        conditions=[c.text.strip() for c in root.findall("condition") if c.text],
        # The registry XML indents the whole block; remove that, keep the bullet structure.
        criteria_text=textwrap.dedent(criteria.replace("\r\n", "\n")).strip(),
        minimum_age=_clean(root.findtext("eligibility/minimum_age")),
        maximum_age=_clean(root.findtext("eligibility/maximum_age")),
        sex=sex.upper() if sex else None,
        overall_status=_clean(root.findtext("overall_status")),
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        fetched_at=SNAPSHOT_DATE,
        source=SOURCE,
    )


def fetch_snapshot_trial(nct_id: str, settings: Settings) -> Trial:
    nct_id = normalize_nct_id(nct_id)
    cache_path = settings.data_dir / "cache" / "trials_trec_snapshot" / f"{nct_id}.json"
    if cache_path.is_file():
        return Trial.model_validate_json(cache_path.read_text(encoding="utf-8"))

    with httpx.Client(timeout=120) as client:
        index = load_index(settings, client)
        if nct_id not in index:
            raise TrialNotFound(f"{nct_id} is not in the TREC {SNAPSHOT_DATE} trial snapshot")
        part, member = index[nct_id]
        with _open_part(settings, part, client) as archive:
            trial = parse_snapshot_xml(archive.read(member))
    if not trial.criteria_text:
        raise TrialNotFound(f"{nct_id} has no eligibility criteria text in the TREC snapshot")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(trial.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return trial
