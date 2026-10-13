import io
import zipfile

import pytest

from trialmatch import trials
from trialmatch.trec_trials import HttpRangeFile, parse_snapshot_xml
from trialmatch.trials import get_trial, run_id_suffix

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<clinical_study>
  <id_info><nct_id>NCT00000102</nct_id></id_info>
  <brief_title>Example Trial</brief_title>
  <overall_status>Completed</overall_status>
  <condition>Glioma</condition>
  <condition>Astrocytoma</condition>
  <eligibility>
    <criteria><textblock>
        Inclusion Criteria:

          -  Age 18 or older

        Exclusion Criteria:

          -  Pregnancy
    </textblock></criteria>
    <gender>Female</gender>
    <minimum_age>18 Years</minimum_age>
    <maximum_age>N/A</maximum_age>
  </eligibility>
</clinical_study>"""


def test_parse_snapshot_xml():
    trial = parse_snapshot_xml(XML)
    assert trial.nct_id == "NCT00000102" and trial.title == "Example Trial"
    assert trial.conditions == ["Glioma", "Astrocytoma"]
    assert trial.criteria_text.startswith("Inclusion Criteria:\n\n  -  Age 18 or older")
    assert trial.sex == "FEMALE" and trial.minimum_age == "18 Years" and trial.maximum_age is None
    assert trial.source == "trec_snapshot_2021-04-27" and trial.fetched_at == "2021-04-27"


def test_source_switch(monkeypatch, settings):
    monkeypatch.setattr(trials, "fetch_trial", lambda nct, s: "from-api")
    monkeypatch.setattr(trials, "fetch_snapshot_trial", lambda nct, s: "from-snapshot")
    assert get_trial("NCT00000102", settings) == "from-api"
    assert get_trial("NCT00000102", settings, "snapshot") == "from-snapshot"
    with pytest.raises(ValueError):
        get_trial("NCT00000102", settings, "elsewhere")
    assert run_id_suffix("api") == "" and run_id_suffix("snapshot") == "__snap2021"


def test_zip_member_can_be_read_through_range_requests():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("part1/NCT0000xxxx/NCT00000102.xml", XML)
        archive.writestr("part1/NCT0000xxxx/NCT00000999.xml", b"<x/>" * 5000)
    data = buffer.getvalue()

    class Response:
        def __init__(self, status_code, content=b"", headers=None):
            self.status_code, self.content, self.headers = status_code, content, headers or {}

        def raise_for_status(self):
            pass

    class Client:
        requests = 0

        def head(self, url, follow_redirects=True):
            return Response(200, headers={"content-length": str(len(data))})

        def get(self, url, headers, follow_redirects=True):
            Client.requests += 1
            start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
            return Response(206, data[start : end + 1])

    with zipfile.ZipFile(HttpRangeFile("http://x/part1.zip", Client(), block=256)) as archive:
        assert archive.read("part1/NCT0000xxxx/NCT00000102.xml") == XML
    assert Client.requests >= 2  # table of contents, then the member
