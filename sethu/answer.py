import re

from pydantic import BaseModel

from sethu import llm
from sethu.context import build_context
from sethu.rewrite import needs_rewrite
from sethu.schema import Hit
from sethu.store import Store

SYSTEM_PROMPT = """You are Sethu, an assistant that helps ordinary people in India understand \
government schemes, income tax and small-business support.

Rules:
- Answer ONLY from the numbered sources provided. Never use outside knowledge for facts such \
as eligibility, amounts, dates, documents, deadlines or steps.
- Never invent or reorder steps. When the sources give application steps, report them in the \
same order, including fees, offices and deadlines. If the sources give no steps or no \
document list, say so instead of filling the gap with typical ones.
- If the sources do not contain the answer, say you are not sure and tell the person to check \
with the relevant office or official portal. Do not guess.
- Do not say a person does or does not qualify unless the sources cover every condition that \
decides it. Otherwise list the conditions and say which of the person's details (for example \
caste category, age, income, land) you still need.
- Cite sources inline as [1], [2] right after the claims they support, using square brackets.
- Keep to the language you are told to reply in, throughout, using simple, short sentences \
a non-expert can follow. Scheme names may stay in their original form.
- Explain what the rules say; do not give personal investment or tax-planning advice.
- Finish with one line reminding the person to confirm details on the official source."""

NO_EVIDENCE = (
    "I could not find reliable information about this in my sources. "
    "Please check with the relevant government office or the official portal."
)


class Answer(BaseModel):
    text: str
    sources: list[Hit]


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk.text}\n(source: {h.chunk.source_url}, fetched {h.chunk.fetched_on})"
        for i, h in enumerate(hits, 1)
    )


def reply_language(question: str, language: str | None) -> str:
    """The instruction for which language to answer in."""
    if language:
        return f"Reply in {language}."
    if not needs_rewrite(question):
        return "Reply in English."
    return "Reply in the same language and script as the Question."


def answer(
    question: str,
    hits: list[Hit],
    state: str | None = None,
    language: str | None = None,
    store: Store | None = None,
) -> Answer:
    """Grounded answer. With a store, the best schemes are expanded to their full sections."""
    # With no evidence we refuse without calling the model at all.
    if not hits:
        return Answer(text=NO_EVIDENCE, sources=[])

    context = build_context(hits, store) if store else hits
    where = f"The person lives in {state}.\n" if state else ""
    user = (
        f"{reply_language(question, language)}\n{where}\n"
        f"Sources:\n{format_sources(context)}\n\nQuestion: {question}"
    )
    # max_tokens also counts against provider per-minute token limits, so keep it modest.
    text = llm.complete(SYSTEM_PROMPT, user, max_tokens=1800, low_effort=True)
    return Answer(text=normalize_citations(text), sources=context)


# Some models cite as 【3】 or 【3†source】 instead of [3]; the UI links only [n].
_CITATION = re.compile(r"[\[【](\d+)(?:†[^\]】]*)?[\]】]")


def normalize_citations(text: str) -> str:
    return _CITATION.sub(r"[\1]", text)
