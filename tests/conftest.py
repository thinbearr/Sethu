import hashlib
import re
from dataclasses import replace
from pathlib import Path

import pytest

import sethu.retrieve as retrieve_module
from sethu.chunking import chunk_record
from sethu.loaders import load_jsonl
from sethu.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


class HashEmbedder:
    """Offline stand-in for the real model: hashed bag-of-words, so tests need no download."""

    DIM = 512

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store(tmp_path, embedder, monkeypatch):
    # Hashed vectors are far less similar than real embeddings, so relax the cutoff.
    monkeypatch.setattr(
        retrieve_module, "settings", replace(retrieve_module.settings, max_distance=0.95)
    )
    s = Store(path=tmp_path / "store")
    for record in load_jsonl(FIXTURES / "schemes.jsonl"):
        chunks = chunk_record(record)
        s.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return s
