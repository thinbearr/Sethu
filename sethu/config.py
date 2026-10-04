import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


EMBEDDING_MODEL = os.getenv("SETHU_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")


@dataclass(frozen=True)
class Settings:
    # BAAI/bge-m3 is better but ~7x slower than e5-base on CPU (4 h vs 30 min for 8.4k chunks).
    embedding_model: str = EMBEDDING_MODEL
    llm_provider: str = os.getenv("SETHU_LLM_PROVIDER", "anthropic")  # "anthropic" or "groq"
    llm_model: str = os.getenv(
        "SETHU_LLM_MODEL",
        "openai/gpt-oss-120b"
        if os.getenv("SETHU_LLM_PROVIDER") == "groq"
        else "claude-sonnet-5-5",
    )
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    # One store per embedding model: vectors from different models must never be mixed.
    store_dir: Path = Path(
        os.getenv("SETHU_STORE_DIR", ROOT / "data" / "store" / EMBEDDING_MODEL.replace("/", "__"))
    )
    collection: str = "sethu"
    top_k: int = int(os.getenv("SETHU_TOP_K", "6"))
    retrieval_mode: str = os.getenv("SETHU_RETRIEVAL_MODE", "hybrid")  # "hybrid" or "dense"
    hybrid_pool: int = int(os.getenv("SETHU_HYBRID_POOL", "30"))  # candidates per ranker
    # Translate non-English questions to an English search query (needs the LLM key).
    rewrite_queries: bool = os.getenv("SETHU_REWRITE", "1") == "1"
    # Chunks farther than this cosine distance are dropped. Measured for e5-base on our eval set:
    # answerable top hits are 0.08-0.21, unrelated questions 0.18-0.25. The ranges overlap, so
    # distance only removes the clearly unrelated; refusing is mainly the LLM's job (see
    # SYSTEM_PROMPT). Re-measure with `python -m sethu.evaluate` if the embedding model changes.
    max_distance: float = float(os.getenv("SETHU_MAX_DISTANCE", "0.25"))
    max_chunk_chars: int = 1200
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")


settings = Settings()
