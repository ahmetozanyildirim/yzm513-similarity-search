"""Standalone naive run for terminal profilers (Phase 1).

    py-spy record -o results/profiling/flamegraph.svg -- python scripts/profile_naive.py
    kernprof -l -v scripts/profile_naive.py          # line_profiler (uses the @profile hook below)
    python -m cProfile -s tottime scripts/profile_naive.py
"""

import builtins

import numpy as np

from hpss import data, naive

if hasattr(builtins, "profile"):                 # defined only when run under kernprof
    naive.cosine_similarity_python = builtins.profile(naive.cosine_similarity_python)
    naive.search_naive = builtins.profile(naive.search_naive)

if __name__ == "__main__":
    X, Q = data.load_embeddings(n=5_000)
    corpus = X.astype(np.float64).tolist()
    queries = Q[:5].astype(np.float64).tolist()
    res = naive.search_naive(queries, corpus, 10)
    print("top-3 of query 0:", res[0][:3])
