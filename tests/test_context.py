from datetime import date

from sethu.answer import NO_EVIDENCE, answer, reply_language
from sethu.chunking import chunk_record
from sethu.context import build_context
from sethu.retrieve import retrieve
from sethu.schema import SchemeRecord


def _add_scheme_with_faq(store, embedder):
    record = SchemeRecord(
        id="faq-scheme",
        name="Test Pond Scheme",
        state="central",
        source_url="https://example.invalid/pond",
        fetched_on=date(2026, 1, 1),
        sections={
            "eligibility": "Owners of land who dig a pond qualify.",
            "application_process": "Step 1 visit the range office. Step 2 pay the fee. Step 3 go to the nursery.",
            "faq": "Can I apply twice? No. " * 40,
        },
    )
    chunks = chunk_record(record)
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_top_scheme_is_expanded_to_all_its_main_sections(store, embedder):
    hits = retrieve("income certificate college admission", store, embedder, state="Karnataka", k=1)
    context = build_context(hits, store)
    sections = {h.chunk.section for h in context if h.chunk.scheme_id == "test-ka-1"}
    assert sections == {"eligibility", "documents"}


def test_application_steps_are_included_in_source_order_but_faq_is_not(store, embedder):
    _add_scheme_with_faq(store, embedder)
    hits = retrieve("pond range office fee nursery", store, embedder, k=1, mode="dense")
    assert hits[0].chunk.scheme_id == "faq-scheme"
    context = build_context(hits, store)
    assert [h.chunk.section for h in context] == ["eligibility", "application_process"]


def test_faq_chunk_is_kept_if_retrieval_itself_picked_it(store, embedder):
    _add_scheme_with_faq(store, embedder)
    hits = retrieve("Can I apply twice", store, embedder, k=3, mode="dense")
    context = build_context(hits, store)
    assert any(h.chunk.section == "faq" for h in context if h.chunk.scheme_id == "faq-scheme")


def test_budget_limits_each_expanded_scheme(store, embedder):
    hits = retrieve("income certificate college admission", store, embedder, k=1)
    context = build_context(hits, store, chars_per_scheme=10)
    assert context == []  # nothing fits, so nothing is invented


def test_other_schemes_are_added_as_a_few_alternatives(store, embedder):
    hits = retrieve("loan grant eligibility", store, embedder, k=20, mode="dense")
    context = build_context(hits, store, full_schemes=1, extra_hits=2)
    others = [h for h in context if h.chunk.scheme_id != context[0].chunk.scheme_id]
    assert len(others) == 2


def test_reply_language_instruction():
    assert reply_language("loan for farmers", None) == "Reply in English."
    assert reply_language("ರೈತರಿಗೆ ಸಾಲ", None) == "Reply in the same language and script as the Question."
    assert reply_language("ರೈತರಿಗೆ ಸಾಲ", "Kannada") == "Reply in Kannada."


def test_no_hits_means_refusal_without_llm(monkeypatch):
    import sethu.answer as a

    monkeypatch.setattr(a.llm, "complete", lambda *x, **k: 1 / 0)
    assert answer("anything", []).text == NO_EVIDENCE
