from pathlib import Path

import chromadb
import numpy as np

from sethu.bm25 import BM25Index
from sethu.config import settings
from sethu.schema import CENTRAL, Chunk, Hit


# Reading order of a scheme's sections when they are shown to the model.
SECTION_ORDER = [
    "details", "eligibility", "benefits", "exclusions",
    "application_process", "documents_required", "faq",
]


def _allowed_states(state: str | None) -> set[str] | None:
    # A state user sees their state's schemes plus the central ones.
    return {state, CENTRAL} if state else None


class Store:
    def __init__(self, path: Path | None = None, collection: str | None = None):
        client = chromadb.PersistentClient(path=str(path or settings.store_dir))
        self.col = client.get_or_create_collection(
            collection or settings.collection, metadata={"hnsw:space": "cosine"}
        )
        self._bm25: BM25Index | None = None
        self._metas: list[dict] = []

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in chunks],
        )
        self._bm25 = None

    def delete_scheme(self, scheme_id: str) -> None:
        """Drop a scheme's old chunks before re-ingesting it, so removed sections don't linger."""
        self.col.delete(where={"scheme_id": scheme_id})
        self._bm25 = None

    def query(
        self,
        embedding: list[float],
        k: int,
        state: str | None = None,
        domain: str | None = None,
    ) -> list[Hit]:
        """Dense nearest-neighbour search."""
        clauses: list[dict] = []
        states = _allowed_states(state)
        if states:
            clauses.append({"state": {"$in": sorted(states)}})
        if domain:
            clauses.append({"domain": domain})
        where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

        res = self.col.query(query_embeddings=[embedding], n_results=k, where=where)
        return [
            Hit(chunk=Chunk(id=id_, text=text, **meta), distance=dist)
            for id_, text, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def lexical(
        self, query: str, k: int, state: str | None = None, domain: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 search with the same state/domain filtering. Returns (chunk id, score)."""
        if self._bm25 is None:
            data = self.col.get(include=["documents", "metadatas"])
            self._bm25 = BM25Index(data["ids"], data["documents"])
            self._metas = data["metadatas"]
        states = _allowed_states(state)
        allowed = {
            i
            for i, m in enumerate(self._metas)
            if (states is None or m["state"] in states) and (domain is None or m["domain"] == domain)
        }
        return self._bm25.search(query, k, allowed)

    def fetch(self, ids: list[str], query_embedding: list[float]) -> dict[str, Hit]:
        """Load chunks by id, with their dense distance to the query."""
        if not ids:
            return {}
        res = self.col.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        q = np.asarray(query_embedding, dtype=float)
        out = {}
        for id_, text, meta, emb in zip(
            res["ids"], res["documents"], res["metadatas"], res["embeddings"]
        ):
            e = np.asarray(emb, dtype=float)
            cos = float(e @ q / (np.linalg.norm(e) * np.linalg.norm(q) or 1.0))
            out[id_] = Hit(chunk=Chunk(id=id_, text=text, **meta), distance=1.0 - cos)
        return out

    def scheme_chunks(self, scheme_id: str) -> list[Chunk]:
        """All chunks of one scheme, in the order they appear in the source."""
        res = self.col.get(where={"scheme_id": scheme_id}, include=["documents", "metadatas"])
        chunks = [
            Chunk(id=id_, text=text, **meta)
            for id_, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        return sorted(chunks, key=lambda c: (SECTION_ORDER.index(c.section) if c.section in SECTION_ORDER else 99, int(c.id.rsplit(":", 1)[1])))

    def count(self) -> int:
        return self.col.count()
