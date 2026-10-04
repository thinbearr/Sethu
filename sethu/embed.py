from typing import Protocol

from sethu.config import settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Multilingual embedder; a Kannada query can match an English document."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = 512  # chunks are ~1200 chars; avoids padding to 8192 on CPU
        # E5 models need role prefixes; BGE-M3 does not.
        e5 = "e5" in self.model_name.lower()
        self._doc_prefix = "passage: " if e5 else ""
        self._query_prefix = "query: " if e5 else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            [self._doc_prefix + t for t in texts], normalize_embeddings=True, batch_size=16
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self._query_prefix + text, normalize_embeddings=True).tolist()
