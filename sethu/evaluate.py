"""Compare retrieval modes on a labelled question set.

Each line of the questions file: {"q", "lang", "state", "expect": [scheme ids], "kind"}.
kind "out_of_scope" has no expected scheme; those measure the refusal cutoff instead.

Ranking quality is measured with the distance cutoff disabled, so it reflects the ranker
alone. The cutoff is judged separately, from the top hit's distance.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import replace

import sethu.retrieve as retrieve_module
from sethu.config import settings
from sethu.embed import SentenceTransformerEmbedder
from sethu.retrieve import retrieve
from sethu.store import Store

DEPTH = 10  # scheme ranks examined


def scheme_ranking(hits):
    seen: list[str] = []
    for h in hits:
        if h.chunk.scheme_id not in seen:
            seen.append(h.chunk.scheme_id)
    return seen


def first_hit_rank(ranking: list[str], expect: list[str]) -> int | None:
    for i, sid in enumerate(ranking, 1):
        if sid in expect:
            return i
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    ranks = [r["rank"] for r in rows]
    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@6": sum(1 for r in ranks if r and r <= 6) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def fmt(s: dict) -> str:
    if not s["n"]:
        return "-"
    return f"n={s['n']:<3} hit@1={s['hit@1']:.2f} hit@3={s['hit@3']:.2f} hit@6={s['hit@6']:.2f} mrr={s['mrr']:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("questions")
    p.add_argument("--modes", nargs="+", default=["dense", "hybrid"])
    p.add_argument("--verbose", action="store_true", help="Print every miss")
    args = p.parse_args()

    items = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    store, embedder = Store(), SentenceTransformerEmbedder()
    print(f"store: {store.count()} chunks | embedder: {settings.embedding_model}\n")

    # Disable the cutoff for ranking metrics; restored below for the refusal analysis.
    real_settings = retrieve_module.settings
    retrieve_module.settings = replace(real_settings, max_distance=float("inf"))

    results: dict[str, list[dict]] = {}
    for mode in args.modes:
        rows = []
        for it in items:
            base, _, flag = mode.partition("+")  # e.g. "hybrid+rewrite"
            hits = retrieve(
                it["q"], store, embedder, it["state"], k=40, mode=base, rewrite=(flag == "rewrite")
            )
            ranking = scheme_ranking(hits)[:DEPTH]
            rows.append(
                {
                    **it,
                    "ranking": ranking,
                    "rank": first_hit_rank(ranking, it["expect"]) if it["expect"] else None,
                    "top_distance": min((h.distance for h in hits), default=None),
                }
            )
        results[mode] = rows
    retrieve_module.settings = real_settings

    for mode, rows in results.items():
        answerable = [r for r in rows if r["expect"]]
        print(f"=== {mode} ===")
        print("all        ", fmt(summarize(answerable)))
        for key in ("kind", "lang"):
            groups = defaultdict(list)
            for r in answerable:
                groups[r[key]].append(r)
            for g in sorted(groups):
                print(f"{key}={g:<9}", fmt(summarize(groups[g])))
        print()

    # Refusal cutoff: where do answerable and out-of-scope top distances fall?
    rows = results[args.modes[0]]
    ans = sorted(r["top_distance"] for r in rows if r["expect"] and r["top_distance"] is not None)
    oos = sorted(r["top_distance"] for r in rows if not r["expect"] and r["top_distance"] is not None)
    print(f"top-hit distance, answerable   : min={ans[0]:.3f} median={ans[len(ans)//2]:.3f} max={ans[-1]:.3f}")
    print(f"top-hit distance, out-of-scope : min={oos[0]:.3f} median={oos[len(oos)//2]:.3f} max={oos[-1]:.3f}")
    print(f"current cutoff (max_distance)  : {settings.max_distance}")
    kept = sum(d <= settings.max_distance for d in ans)
    refused = sum(d > settings.max_distance for d in oos)
    print(f"at this cutoff: {kept}/{len(ans)} answerable kept, {refused}/{len(oos)} out-of-scope refused")

    if args.verbose:
        for mode, rows in results.items():
            print(f"\n--- misses ({mode}) ---")
            for r in rows:
                if r["expect"] and (r["rank"] is None or r["rank"] > 3):
                    print(f"[{r['lang']}] rank={r['rank']} q={r['q'][:70]!r}\n    expect={r['expect']} got={r['ranking'][:4]}")


if __name__ == "__main__":
    main()
