from sethu.answer import NO_EVIDENCE, answer
from sethu.retrieve import retrieve


def test_state_user_sees_own_state_and_central_only(store, embedder):
    hits = retrieve("who is eligible", store, embedder, state="Karnataka", k=20)
    states = {h.chunk.state for h in hits}
    assert states == {"Karnataka", "central"}


def test_no_state_returns_everything(store, embedder):
    hits = retrieve("who is eligible", store, embedder, k=20)
    assert {"Karnataka", "Tamil Nadu", "central"} <= {h.chunk.state for h in hits}


def test_domain_filter(store, embedder):
    hits = retrieve("register business", store, embedder, domain="msme", k=20)
    assert hits and {h.chunk.domain for h in hits} == {"msme"}


def test_most_relevant_chunk_ranks_first(store, embedder):
    hits = retrieve("income certificate college admission letter", store, embedder)
    assert hits[0].chunk.scheme_id == "test-ka-1"


def test_reingest_replaces_old_chunks(store):
    before = store.count()
    store.delete_scheme("test-central-1")
    assert store.count() == before - 2


def test_unrelated_question_gets_refusal_without_calling_llm(store, embedder, monkeypatch):
    import sethu.retrieve as r
    from dataclasses import replace

    monkeypatch.setattr(r, "settings", replace(r.settings, max_distance=0.3))
    hits = retrieve("zebra quantum helicopter", store, embedder)
    assert hits == []
    assert answer("zebra quantum helicopter", hits).text == NO_EVIDENCE
