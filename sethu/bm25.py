import math
import re
from collections import Counter, defaultdict

# \w misses Indic vowel signs (they are combining marks), so a Hindi or Tamil word would be
# cut into pieces. Adding the Devanagari..Malayalam block range keeps whole words together.
_TOKEN = re.compile(r"[\wऀ-෿]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of documents, with an inverted index."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.lens.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((i, n))
        n_docs = len(ids)
        self.avgdl = (sum(self.lens) / n_docs) if n_docs else 1.0
        self.idf = {
            t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def search(
        self, query: str, k: int, allowed: set[int] | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (id, score), best first. `allowed` restricts results to those document indexes."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[i], s) for i, s in top]
