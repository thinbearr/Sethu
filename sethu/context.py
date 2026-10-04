from sethu.schema import Hit
from sethu.store import Store

# FAQ chunks are only used when retrieval itself picked them: they are long and often repeat
# the other sections, but they would crowd out the application steps if added wholesale.
SKIP_WHEN_EXPANDING = {"faq"}


def build_context(
    hits: list[Hit],
    store: Store,
    full_schemes: int = 2,
    chars_per_scheme: int = 5000,
    extra_hits: int = 3,
) -> list[Hit]:
    """Widen retrieved chunks into the context the model sees.

    Retrieval finds the right scheme but returns fragments, and a model asked for the
    application steps will improvise any it was not shown. So the best `full_schemes` schemes
    are included with their main sections (up to `chars_per_scheme` each), plus up to
    `extra_hits` retrieved chunks from other schemes as alternatives.
    """
    schemes: list[str] = []
    for h in hits:
        if h.chunk.scheme_id not in schemes:
            schemes.append(h.chunk.scheme_id)
    chosen = schemes[:full_schemes]

    best = {}
    for h in hits:
        best.setdefault(h.chunk.scheme_id, h)

    context: list[Hit] = []
    for scheme_id in chosen:
        used = 0
        retrieved_ids = {h.chunk.id for h in hits if h.chunk.scheme_id == scheme_id}
        for chunk in store.scheme_chunks(scheme_id):
            if chunk.section in SKIP_WHEN_EXPANDING and chunk.id not in retrieved_ids:
                continue
            if used + len(chunk.text) > chars_per_scheme:
                continue
            used += len(chunk.text)
            context.append(Hit(chunk=chunk, distance=best[scheme_id].distance))

    others = [h for h in hits if h.chunk.scheme_id not in chosen][:extra_hits]
    return context + others
