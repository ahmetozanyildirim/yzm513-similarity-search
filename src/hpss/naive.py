"""Phase 1 - the naive baseline (pure Python, intentionally slow).

`search_naive` is the reference implementation: every faster version in this
project must return the same top-k (within 1e-5).

The bottom half contains "Python-level fixes": still pure Python, but with the
algorithmic waste removed. They show how far you get *without* leaving the
interpreter - and that the interpreter itself is the real bottleneck.
"""

from __future__ import annotations

import heapq
import math
from operator import mul


def cosine_similarity_python(a, b):
    """Cosine similarity of two equal-length lists of floats. No NumPy."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search_naive(queries, corpus, k=10):
    """For each query: score every document, sort all scores, keep the top-k.

    queries, corpus: lists of lists of floats
    returns: [[(doc_idx, score), ...k items], ...one list per query]
    """
    results = []
    for query in queries:
        scores = []
        for d_idx, doc in enumerate(corpus):
            sim = cosine_similarity_python(query, doc)
            scores.append((d_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


# ----------------------------------------------------------------------
# Python-level fixes (still no NumPy) - used to test the bottleneck hypotheses
# ----------------------------------------------------------------------

def normalize_python(vectors):
    """Divide every vector by its length once, so cosine == dot product."""
    out = []
    for v in vectors:
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n > 0 else list(v))
    return out


def search_python_prenorm(queries, corpus_normed, k=10):
    """Fix #1: norms computed once (not 2x per pair) -> 1/3 of the arithmetic."""
    results = []
    for query in normalize_python(queries):
        scores = []
        for d_idx, doc in enumerate(corpus_normed):
            dot = 0.0
            for i in range(len(query)):
                dot += query[i] * doc[i]
            scores.append((d_idx, dot))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


def search_python_builtin(queries, corpus_normed, k=10):
    """Fix #2 + #3: dot product with sum(map(mul, ...)) (the loop runs in C inside
    the built-ins) and heapq.nlargest instead of sorting all N scores."""
    results = []
    for query in normalize_python(queries):
        scores = [sum(map(mul, query, doc)) for doc in corpus_normed]
        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        results.append([(i, scores[i]) for i in top])
    return results
