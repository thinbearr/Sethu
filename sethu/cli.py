import argparse

from sethu.answer import answer
from sethu.chunking import chunk_record
from sethu.config import settings
from sethu.embed import SentenceTransformerEmbedder
from sethu.loaders import load_jsonl
from sethu.retrieve import retrieve
from sethu.store import Store


def cmd_ingest(args: argparse.Namespace) -> None:
    store, embedder = Store(), SentenceTransformerEmbedder()
    n_records = n_chunks = 0
    wanted = {s.strip() for s in args.states.split(",")} | {"central"} if args.states else None
    for record in load_jsonl(args.path):
        if wanted and record.state not in wanted:
            continue
        chunks = chunk_record(record)
        if not chunks:
            continue
        store.delete_scheme(record.id)
        store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
        n_records += 1
        n_chunks += len(chunks)
        if n_records % 25 == 0:
            print(f"  {n_records} records, {n_chunks} chunks", flush=True)
    print(f"Ingested {n_records} records as {n_chunks} chunks. Store now holds {store.count()}.")


def cmd_ask(args: argparse.Namespace) -> None:
    store = Store()
    hits = retrieve(
        args.question, store, SentenceTransformerEmbedder(), args.state, args.domain, mode=args.mode
    )
    key = settings.groq_api_key if settings.llm_provider == "groq" else settings.anthropic_api_key
    if not key:
        print(f"(no API key for provider '{settings.llm_provider}': showing retrieved sources only)\n")
        for i, h in enumerate(hits, 1):
            ranks = f"dense#{h.dense_rank} lexical#{h.lexical_rank}"
            print(f"[{i}] distance={h.distance:.3f} ({ranks})\n{h.chunk.text}\n  {h.chunk.source_url}\n")
        return
    result = answer(args.question, hits, args.state, args.language, store)
    print(result.text)
    for i, h in enumerate(result.sources, 1):
        print(f"[{i}] {h.chunk.scheme_name}: {h.chunk.source_url} (fetched {h.chunk.fetched_on})")


def main() -> None:
    p = argparse.ArgumentParser(prog="sethu")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest a normalized JSONL file")
    ing.add_argument("path")
    ing.add_argument("--states", help="Comma-separated states to ingest (central is always included)")
    ing.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--state")
    ask.add_argument("--language", help='Language to answer in, e.g. "Kannada" (default: match the question)')
    ask.add_argument("--domain", choices=["scheme", "tax", "msme"])
    ask.add_argument("--mode", choices=["dense", "hybrid"], help="Default: SETHU_RETRIEVAL_MODE")
    ask.set_defaults(func=cmd_ask)

    args = p.parse_args()
    args.func(args)
