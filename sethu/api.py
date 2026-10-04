import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from sethu.answer import answer
from sethu.embed import SentenceTransformerEmbedder
from sethu.retrieve import retrieve
from sethu.store import Store
from sethu.tts import synthesize

app = FastAPI(title="Sethu")
WEB = Path(__file__).parent / "web"


class AskRequest(BaseModel):
    question: str
    state: str | None = None
    domain: Literal["scheme", "tax", "msme"] | None = None
    mode: Literal["dense", "hybrid"] | None = None
    language: str | None = None  # language to answer in; default matches the question


class SourceRef(BaseModel):
    n: int  # the [n] used in the answer text
    scheme_name: str
    state: str
    section: str
    source_url: str
    fetched_on: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]  # only the sources the answer actually cites


@lru_cache
def _embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@lru_cache
def _store() -> Store:
    return Store()


@app.get("/")
def index() -> FileResponse:
    # no-cache = revalidate every time (cheap, via ETag), so a UI fix reaches users at once
    # instead of browsers heuristically reusing an old copy.
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": _store().count()}


class SpeakRequest(BaseModel):
    text: str
    language: str | None = None  # only used to tell Marathi from Hindi


@app.post("/tts")
async def tts(req: SpeakRequest) -> Response:
    hint = "mr" if (req.language or "").lower().startswith("marathi") else None
    try:
        audio = await synthesize(req.text, hint)
    except ValueError as e:
        raise HTTPException(400, "There is nothing to read aloud.") from e
    except Exception as e:  # the speech service is external and can be down
        raise HTTPException(503, "Read-aloud is unavailable right now.") from e
    return Response(audio, media_type="audio/mpeg")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > 1000:
        raise HTTPException(400, "Question is too long.")
    try:
        hits = retrieve(question, _store(), _embedder(), req.state, req.domain, mode=req.mode)
        result = answer(question, hits, req.state, req.language, _store())
    except Exception as e:  # provider outage or rate limit: a friendly retry, not a stack trace
        raise HTTPException(503, "The assistant is busy right now. Please try again in a minute.") from e

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result.text)}
    sources = [
        SourceRef(
            n=i,
            scheme_name=h.chunk.scheme_name,
            state=h.chunk.state,
            section=h.chunk.section,
            source_url=h.chunk.source_url,
            fetched_on=h.chunk.fetched_on,
        )
        for i, h in enumerate(result.sources, 1)
        if i in cited
    ]
    return AskResponse(answer=result.text, sources=sources)
