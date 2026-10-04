import json
from collections.abc import Iterator
from pathlib import Path

from sethu.schema import SchemeRecord


def load_jsonl(path: str | Path) -> Iterator[SchemeRecord]:
    """Read normalized records, one JSON object per line.

    Source-specific adapters (myScheme, PDFs, ...) should convert their data
    into this format, so ingestion never depends on a source's raw layout.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield SchemeRecord.model_validate(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
