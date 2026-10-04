import sethu.retrieve as retrieve_module
from sethu import rewrite
from sethu.retrieve import retrieve


def test_needs_rewrite_only_for_non_latin_text():
    assert not rewrite.needs_rewrite("loan for farmers")
    assert not rewrite.needs_rewrite("PM-KISAN 2024?")
    assert rewrite.needs_rewrite("ರೈತರಿಗೆ ಸಾಲ")
    assert rewrite.needs_rewrite("किसान योजना PM-KISAN")
    assert not rewrite.needs_rewrite("")


def test_failure_is_not_cached_so_a_retry_can_succeed(monkeypatch):
    rewrite._cache.clear()
    monkeypatch.setattr(rewrite.llm, "has_key", lambda: True)
    monkeypatch.setattr(rewrite.llm, "complete", lambda *a, **k: 1 / 0)
    assert rewrite.rewrite_query("ರೈತರಿಗೆ ಸಾಲ") is None
    monkeypatch.setattr(rewrite.llm, "complete", lambda *a, **k: "loan for farmers")
    assert rewrite.rewrite_query("ರೈತರಿಗೆ ಸಾಲ") == "loan for farmers"


def test_citation_styles_are_normalised():
    from sethu.answer import normalize_citations

    assert normalize_citations("a 【3】 b [1†source][2†source] c [4]") == "a [3] b [1][2] c [4]"
    assert normalize_citations("price [₹5] and 【7†L12-L14】") == "price [₹5] and [7]"


def test_rewrite_returns_none_without_calling_llm_for_english(monkeypatch):
    rewrite._cache.clear()
    monkeypatch.setattr(rewrite.llm, "has_key", lambda: True)
    monkeypatch.setattr(rewrite.llm, "complete", lambda *a, **k: 1 / 0)  # must not be called
    assert rewrite.rewrite_query("loan for farmers") is None


def test_rewrite_failure_is_swallowed(monkeypatch):
    rewrite._cache.clear()
    monkeypatch.setattr(rewrite.llm, "has_key", lambda: True)
    monkeypatch.setattr(rewrite.llm, "complete", lambda *a, **k: 1 / 0)
    assert rewrite.rewrite_query("ರೈತರಿಗೆ ಸಾಲ") is None


def test_rewrite_strips_quotes_and_skips_without_key(monkeypatch):
    rewrite._cache.clear()
    monkeypatch.setattr(rewrite.llm, "has_key", lambda: False)
    assert rewrite.rewrite_query("ರೈತರಿಗೆ ಸಾಲ") is None
    rewrite._cache.clear()
    monkeypatch.setattr(rewrite.llm, "has_key", lambda: True)
    monkeypatch.setattr(rewrite.llm, "complete", lambda *a, **k: ' "loan for farmers" ')
    assert rewrite.rewrite_query("ರೈತರಿಗೆ ಸಾಲ") == "loan for farmers"


def test_english_rewrite_rescues_a_query_the_embedder_cannot_read(store, embedder, monkeypatch):
    # The hash embedder shares no tokens with Kannada text, like a weak multilingual model.
    kannada = "ವಿದ್ಯಾರ್ಥಿ ಅನುದಾನ"
    monkeypatch.setattr(
        retrieve_module, "rewrite_query", lambda q: "student grant college admission letter"
    )
    without = retrieve(kannada, store, embedder, state="Karnataka", rewrite=False, mode="hybrid")
    with_rw = retrieve(kannada, store, embedder, state="Karnataka", rewrite=True, mode="hybrid")
    assert not without or without[0].chunk.scheme_id != "test-ka-1"
    assert with_rw[0].chunk.scheme_id == "test-ka-1"
    assert with_rw[0].lexical_rank == 1  # BM25 ran on the English text


def test_rewrite_off_never_calls_rewriter(store, embedder, monkeypatch):
    monkeypatch.setattr(retrieve_module, "rewrite_query", lambda q: 1 / 0)
    assert retrieve("farmer land", store, embedder, rewrite=False)
