from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["scheme", "tax", "msme"]

CENTRAL = "central"


class SchemeRecord(BaseModel):
    """One normalized document, whatever source it came from."""

    id: str
    name: str
    state: str = CENTRAL  # "central" or a state name, e.g. "Karnataka"
    domain: Domain = "scheme"
    language: str = "en"
    source_url: str
    fetched_on: date
    # section title -> text, e.g. {"eligibility": "...", "documents": "..."}
    sections: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    scheme_id: str
    scheme_name: str
    state: str
    domain: str
    language: str
    section: str
    source_url: str
    fetched_on: str

    def metadata(self) -> dict[str, str]:
        """Everything except id/text, in the flat form the vector store needs."""
        return self.model_dump(exclude={"id", "text"})


class Hit(BaseModel):
    chunk: Chunk
    distance: float  # dense cosine distance to the query; always set, even for lexical-only hits
    score: float | None = None  # fused score (RRF) in hybrid mode
    dense_rank: int | None = None  # 1-based position in each ranking, None if absent
    lexical_rank: int | None = None
