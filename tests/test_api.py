from datetime import date

from fastapi.testclient import TestClient

import sethu.api as api
from sethu.answer import Answer
from sethu.schema import Chunk, Hit


def _hit(section="eligibility", scheme="Test Scheme"):
    chunk = Chunk(
        id=f"s:{section}:0", text="x", scheme_id="s", scheme_name=scheme, state="Karnataka",
        domain="scheme", language="en", section=section,
        source_url="https://example.invalid/s", fetched_on=date(2026, 1, 1).isoformat(),
    )
    return Hit(chunk=chunk, distance=0.1)


def _client(monkeypatch, answer_text):
    monkeypatch.setattr(api, "_store", lambda: None)
    monkeypatch.setattr(api, "_embedder", lambda: None)
    monkeypatch.setattr(api, "retrieve", lambda *a, **k: [_hit()])
    monkeypatch.setattr(
        api, "answer", lambda *a, **k: Answer(text=answer_text, sources=[_hit("benefits"), _hit("eligibility", "Other")])
    )
    return TestClient(api.app)


def test_only_cited_sources_are_returned(monkeypatch):
    r = _client(monkeypatch, "You get money [2].").post("/ask", json={"question": "help?"})
    assert r.status_code == 200
    body = r.json()
    assert [s["n"] for s in body["sources"]] == [2]
    assert body["sources"][0]["scheme_name"] == "Other"


def test_empty_and_oversized_questions_are_rejected(monkeypatch):
    c = _client(monkeypatch, "x")
    assert c.post("/ask", json={"question": "   "}).status_code == 400
    assert c.post("/ask", json={"question": "a" * 1001}).status_code == 400


def test_backend_failure_becomes_friendly_503(monkeypatch):
    c = _client(monkeypatch, "x")
    monkeypatch.setattr(api, "retrieve", lambda *a, **k: 1 / 0)
    r = c.post("/ask", json={"question": "help?"})
    assert r.status_code == 503
    assert "try again" in r.json()["detail"]


def test_index_page_is_served_and_must_be_revalidated(monkeypatch):
    r = _client(monkeypatch, "x").get("/")
    assert r.status_code == 200 and "Sethu" in r.text
    assert r.headers["cache-control"] == "no-cache"
