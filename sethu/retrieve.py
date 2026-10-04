from sethu.config import settings
from sethu.embed import Embedder
from sethu.rewrite import rewrite_query
from sethu.schema import Hit
from sethu.store import Store

RRF_C = 60  # standard constant; damps the advantage of rank 1 over rank 2


def rrf(rankings: list[list[str]], c: int = RRF_C) -> dict[str, float]:
    """Reciprocal rank fusion: each ranking adds 1/(c + rank) to an id's score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, 1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (c + rank)
    return scores


def _keep_closest(by_id: dict[str, Hit], hits) -> None:
    for h in hits:
        if h.chunk.id not in by_id or h.distance < by_id[h.chunk.id].distance:
            by_id[h.chunk.id] = h


def retrieve(
    question: str,
    store: Store,
    embedder: Embedder,
    state: str | None = None,
    domain: str | None = None,
    k: int | None = None,
    mode: str | None = None,
    rewrite: bool | None = None,
) -> list[Hit]:
    """Top chunks for a question, minus anything too far (densely) to count as evidence.

    mode "dense": vector search only. mode "hybrid": vector search + BM25.
    With rewrite on, a non-English question is also searched as an English query, because
    the documents are English; BM25 only works on that English text. Rankings are fused
    with RRF. A chunk's distance is its smallest dense distance over the queries, so the
    evidence cutoff cannot be passed by a lexical match alone.
    """
    k = k or settings.top_k
    mode = mode or settings.retrieval_mode
    if mode not in ("dense", "hybrid"):
        raise ValueError(f"unknown retrieval mode: {mode!r}")
    use_rewrite = settings.rewrite_queries if rewrite is None else rewrite

    english = rewrite_query(question) if use_rewrite else None
    texts = [question] + ([english] if english else [])
    vecs = [embedder.embed_query(t) for t in texts]
    pool = k if mode == "dense" else max(settings.hybrid_pool, k)

    dense_rankings: list[list[str]] = []
    by_id: dict[str, Hit] = {}
    for vec in vecs:
        hits = store.query(vec, pool, state, domain)
        dense_rankings.append([h.chunk.id for h in hits])
        _keep_closest(by_id, hits)

    lexical_ids: list[str] = []
    if mode == "hybrid":
        # The English rewrite when there is one, since the documents are English.
        lexical_ids = [i for i, _ in store.lexical(texts[-1], pool, state, domain)]

    rankings = dense_rankings + ([lexical_ids] if lexical_ids else [])
    fused = rrf(rankings)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:k]

    missing = [i for i in top_ids if i not in by_id]
    for vec in vecs:  # lexical-only hits still get a real dense distance
        _keep_closest(by_id, store.fetch(missing, vec).values())

    out = []
    for id_ in top_ids:
        hit = by_id[id_]
        hit.score = fused[id_]
        ranks = [r.index(id_) + 1 for r in dense_rankings if id_ in r]
        hit.dense_rank = min(ranks) if ranks else None
        hit.lexical_rank = lexical_ids.index(id_) + 1 if id_ in lexical_ids else None
        out.append(hit)
    return [h for h in out if h.distance <= settings.max_distance]
