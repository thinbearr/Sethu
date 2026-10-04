import asyncio

import pytest
from fastapi.testclient import TestClient

import sethu.api as api
from sethu import tts


def test_language_is_detected_from_script():
    assert tts.detect_language("ಹೌದು, ಯೋಜನೆ ಲಭ್ಯವಿದೆ") == "kn"
    assert tts.detect_language("అవును, పథకం ఉంది") == "te"
    assert tts.detect_language("ஆம், திட்டம் உள்ளது") == "ta"
    assert tts.detect_language("അതെ, പദ്ധതി ലഭ്യമാണ്") == "ml"
    assert tts.detect_language("হ্যাঁ, প্রকল্প আছে") == "bn"
    assert tts.detect_language("हाँ, योजना उपलब्ध है") == "hi"
    assert tts.detect_language("Yes, the scheme is available") == "en"


def test_marathi_hint_only_breaks_the_devanagari_tie():
    assert tts.detect_language("हाँ, योजना आहे", hint="mr") == "mr"
    assert tts.detect_language("Yes it is", hint="mr") == "en"
    assert tts.detect_language("ಹೌದು", hint="mr") == "kn"


def test_every_language_has_a_voice():
    assert {"en", "hi", "kn", "te", "ta", "ml", "bn", "mr"} == set(tts.VOICES)


def test_speech_text_drops_citations_markdown_and_urls():
    out = tts.clean_for_speech("**Yes** [1][2]\n- Apply at https://example.com/x now.\n# Note")
    assert "[" not in out and "*" not in out and "http" not in out and "#" not in out
    assert "Yes" in out and "Apply at" in out


class _FakeStream:
    calls: list = []

    def __init__(self, text, voice):
        _FakeStream.calls.append((text, voice))

    async def stream(self):
        yield {"type": "WordBoundary", "data": b"ignored"}
        yield {"type": "audio", "data": b"ID3"}
        yield {"type": "audio", "data": b"abc"}


def test_synthesize_joins_audio_chunks_picks_voice_and_caches(monkeypatch):
    tts._cache.clear()
    _FakeStream.calls = []
    monkeypatch.setattr(tts.edge_tts, "Communicate", _FakeStream)
    audio = asyncio.run(tts.synthesize("ಹೌದು [1]"))
    assert audio == b"ID3abc"
    assert _FakeStream.calls == [("ಹೌದು", "kn-IN-SapnaNeural")]
    asyncio.run(tts.synthesize("ಹೌದು [1]"))  # same cleaned text: served from cache
    assert len(_FakeStream.calls) == 1


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        asyncio.run(tts.synthesize("[1] ** "))


def test_tts_endpoint_returns_audio_and_friendly_errors(monkeypatch):
    client = TestClient(api.app)

    async def ok(text, hint=None):
        return b"MP3DATA"

    monkeypatch.setattr(api, "synthesize", ok)
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 200 and r.content == b"MP3DATA" and r.headers["content-type"] == "audio/mpeg"

    async def down(text, hint=None):
        raise RuntimeError("service down")

    monkeypatch.setattr(api, "synthesize", down)
    assert client.post("/tts", json={"text": "hello"}).status_code == 503

    async def empty(text, hint=None):
        raise ValueError("nothing to read")

    monkeypatch.setattr(api, "synthesize", empty)
    assert client.post("/tts", json={"text": "[1]"}).status_code == 400
