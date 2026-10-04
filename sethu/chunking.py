import re

from sethu.config import settings
from sethu.schema import Chunk, SchemeRecord

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def _split(text: str, max_chars: int) -> list[str]:
    """Split on paragraphs, then sentences, packing pieces up to max_chars."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        pieces.extend(_SENTENCE_END.split(para.strip()) if len(para) > max_chars else [para.strip()])

    out: list[str] = []
    buf = ""
    for piece in filter(None, pieces):
        if buf and len(buf) + len(piece) + 1 > max_chars:
            out.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n{piece}" if buf else piece
    if buf:
        out.append(buf)
    # A single sentence longer than max_chars is hard-cut rather than dropped.
    return [p[i : i + max_chars] for p in out for i in range(0, len(p), max_chars)]


def chunk_record(record: SchemeRecord, max_chars: int | None = None) -> list[Chunk]:
    """Chunk each section separately, prefixing the scheme name so every chunk stands alone."""
    max_chars = max_chars or settings.max_chunk_chars
    chunks: list[Chunk] = []
    for section, body in record.sections.items():
        if not body.strip():
            continue
        header = f"{record.name} ({record.state}) - {section}\n"
        for i, part in enumerate(_split(body, max_chars - len(header))):
            chunks.append(
                Chunk(
                    id=f"{record.id}:{section}:{i}",
                    text=header + part,
                    scheme_id=record.id,
                    scheme_name=record.name,
                    state=record.state,
                    domain=record.domain,
                    language=record.language,
                    section=section,
                    source_url=record.source_url,
                    fetched_on=record.fetched_on.isoformat(),
                )
            )
    return chunks
