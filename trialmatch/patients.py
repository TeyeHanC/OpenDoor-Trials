"""Patient note sources.

Only synthetic or publicly released patient descriptions are supported:

* ``sample-*``     hand-written synthetic notes bundled with the package
* ``trec2021-N``   TREC Clinical Trials 2021 topic N (synthetic case descriptions)
* ``trec2022-N``   TREC Clinical Trials 2022 topic N

There is deliberately no loader for arbitrary files or for restricted corpora
such as MIMIC: every note this module returns may be sent to an external API.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from .config import Settings
from .models import Patient

SAMPLES_DIR = Path(__file__).parent / "samples"
TREC_YEARS = (2021, 2022)
TOPICS_URL = "https://www.trec-cds.org/topics{year}.xml"
QRELS_URL = "https://trec.nist.gov/data/trials/qrels{year}.txt"
PATIENT_ID_RE = re.compile(r"^trec(\d{4})-(\d+)$")
NOTE_DATE_RE = re.compile(r"^Note date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


class PatientNotFound(Exception):
    pass


def topics_path(settings: Settings, year: int) -> Path:
    return settings.trec_dir / f"topics{year}.xml"


def qrels_path(settings: Settings, year: int) -> Path:
    return settings.trec_dir / f"qrels{year}.txt"


def fetch_trec(settings: Settings, year: int, *, refresh: bool = False) -> None:
    """Download the TREC CT topics (patients) and qrels (relevance labels)."""
    if year not in TREC_YEARS:
        raise ValueError(f"TREC Clinical Trials year must be one of {TREC_YEARS}")
    settings.trec_dir.mkdir(parents=True, exist_ok=True)
    for url, path in (
        (TOPICS_URL.format(year=year), topics_path(settings, year)),
        (QRELS_URL.format(year=year), qrels_path(settings, year)),
    ):
        if path.is_file() and not refresh:
            continue
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)


def load_trec_topics(settings: Settings, year: int) -> dict[int, str]:
    path = topics_path(settings, year)
    if not path.is_file():
        raise PatientNotFound(
            f"TREC {year} topics are not downloaded yet. Run: python -m trialmatch fetch-trec --year {year}"
        )
    root = ET.parse(path).getroot()
    return {int(topic.attrib["number"]): (topic.text or "").strip() for topic in root.iter("topic")}


def load_qrels(settings: Settings, year: int) -> list[tuple[int, str, int]]:
    """Return (topic, nct_id, label). Labels: 0 not relevant, 1 excluded, 2 eligible."""
    path = qrels_path(settings, year)
    if not path.is_file():
        raise FileNotFoundError(
            f"TREC {year} qrels are not downloaded yet. Run: python -m trialmatch fetch-trec --year {year}"
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows.append((int(parts[0]), parts[2], int(parts[3])))
    return rows


def _load_sample(patient_id: str) -> Patient:
    path = SAMPLES_DIR / f"{patient_id}.txt"
    if not path.is_file():
        raise PatientNotFound(f"No bundled sample note named {patient_id!r}")
    text = path.read_text(encoding="utf-8").strip()
    match = NOTE_DATE_RE.search(text)
    return Patient(
        id=patient_id,
        source="Synthetic sample (hand-written, not a real patient)",
        text=text,
        note_date=match.group(1) if match else None,
    )


def load_patient(patient_id: str, settings: Settings) -> Patient:
    patient_id = patient_id.strip()
    if patient_id.startswith("sample-"):
        return _load_sample(patient_id)
    match = PATIENT_ID_RE.match(patient_id)
    if not match:
        raise PatientNotFound(
            f"Unknown patient id {patient_id!r}. Use sample-<name> or trec<year>-<topic number>."
        )
    year, number = int(match.group(1)), int(match.group(2))
    topics = load_trec_topics(settings, year)
    if number not in topics:
        raise PatientNotFound(f"TREC {year} has no topic {number}")
    return Patient(id=patient_id, source=f"TREC Clinical Trials {year}, topic {number}", text=topics[number])


def list_patients(settings: Settings) -> list[dict]:
    """IDs offered in the UI: bundled samples plus any downloaded TREC topics."""
    patients = [
        {"id": path.stem, "label": f"{path.stem} (synthetic sample)"}
        for path in sorted(SAMPLES_DIR.glob("sample-*.txt"))
    ]
    for year in TREC_YEARS:
        if topics_path(settings, year).is_file():
            for number, text in sorted(load_trec_topics(settings, year).items()):
                patients.append({"id": f"trec{year}-{number}", "label": f"TREC {year} #{number}: {text[:70]}…"})
    return patients
