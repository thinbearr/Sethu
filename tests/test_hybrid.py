from sethu.bm25 import BM25Index, tokenize
from sethu.chunking import chunk_record
from sethu.loaders import load_jsonl
from sethu.retrieve import retrieve, rrf
from sethu.store import Store

from conftest import FIXTURES


class BlindEmbedder:
    """Every text gets the same vector, so dense search carries no signal at all."""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_tokenizer_keeps_indic_words_whole():
    assert tokenize("किसान योजना") == ["किसान", "योजना"]
    assert tokenize("ರೈತರ ಯೋಜನೆ") == ["ರೈತರ", "ಯೋಜನೆ"]
    assert tokenize("PM-KISAN, AABY!") == ["pm", "kisan", "aaby"]


def test_bm25_prefers_rare_term_and_respects_allowed_set():
    idx = BM25Index(["a", "b", "c"], ["loan for farmers", "loan for students", "loan loan zebra"])
    assert idx.search("zebra", 3)[0][0] == "c"
    assert {i for i, _ in idx.search("loan", 3, allowed={0, 1})} == {"a", "b"}
    assert idx.search("nonexistent", 3) == []


def test_rrf_rewards_agreement_between_rankers():
    scores = rrf([["x", "y", "z"], ["y", "q"]])
    assert max(scores, key=scores.get) == "y"  # second in one list, first in the other
    assert set(scores) == {"x", "y", "z", "q"}


def test_hybrid_finds_exact_name_when_dense_is_blind(tmp_path):
    emb, store = BlindEmbedder(), Store(path=tmp_path / "blind")
    for record in load_jsonl(FIXTURES / "schemes.jsonl"):
        chunks = chunk_record(record)
        store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))

    hits = retrieve("Micro Business Registration", store, emb, k=1, mode="hybrid")
    assert hits[0].chunk.scheme_id == "test-msme-1"
    assert hits[0].lexical_rank == 1


def test_lexical_only_hit_gets_a_real_dense_distance(store, embedder):
    q = embedder.embed_query("income certificate college")
    dense = {h.chunk.id: h.distance for h in store.query(q, 50)}
    fetched = store.fetch(list(dense), q)
    assert all(abs(fetched[i].distance - d) < 1e-4 for i, d in dense.items())


def test_hybrid_respects_state_filter(store, embedder):
    hits = retrieve("Tamil Nadu loan", store, embedder, state="Karnataka", k=20, mode="hybrid")
    assert {h.chunk.state for h in hits} <= {"Karnataka", "central"}


def test_dense_mode_still_works_and_unknown_mode_fails(store, embedder):
    assert retrieve("farmer land", store, embedder, mode="dense")
    import pytest

    with pytest.raises(ValueError):
        retrieve("x", store, embedder, mode="nope")
