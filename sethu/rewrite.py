import logging

from sethu import llm

log = logging.getLogger("sethu.rewrite")

SYSTEM = (
    "You turn a user's question about Indian government schemes into a short English search "
    "query for a database of scheme documents written in English. Keep scheme names, "
    "acronyms, state names, amounts and eligibility details. Output only the query, no quotes."
)

_cache: dict[str, str] = {}  # successes only: a transient failure must not stick


def needs_rewrite(text: str) -> bool:
    """True when the text is mostly non-ASCII, i.e. not English written in Latin letters."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(ord(c) > 127 for c in letters) / len(letters) > 0.3


def rewrite_query(question: str) -> str | None:
    """English search query for a non-English question, or None if not needed or on failure.

    Retrieval must never break because the rewrite did, so errors are logged, not raised.
    A failure makes search fall back to the original question, which is measurably worse for
    some languages, hence the warning.
    """
    if not needs_rewrite(question) or not llm.has_key():
        return None
    if question in _cache:
        return _cache[question]
    try:
        out = llm.complete(SYSTEM, question, max_tokens=400, low_effort=True).strip()
    except Exception as e:
        log.warning("query rewrite failed, searching with the original question: %s", e)
        return None
    out = out.strip("\"'` ")
    if out:
        _cache[question] = out
    return out or None
