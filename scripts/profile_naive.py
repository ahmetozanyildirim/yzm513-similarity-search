"""Standalone naive run for terminal profilers (Phase 1).

    py-spy record -o results/profiling/flamegraph.svg -- python scripts/profile_naive.py
    kernprof -l -v scripts/profile_naive.py          # line_profiler (uses the @profile hook below)
    python scripts/profile_naive.py --cprofile out.prof   # cProfile of search_naive only (used by exp1)
"""

import argparse
import builtins
import cProfile
import time

import numpy as np

from hpss import config, data, naive

if hasattr(builtins, "profile"):                 # defined only when run under kernprof
    naive.cosine_similarity_python = builtins.profile(naive.cosine_similarity_python)
    naive.search_naive = builtins.profile(naive.search_naive)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cprofile", metavar="OUT.prof", help="profile search_naive with cProfile, dump the stats here")
    args = ap.parse_args()
    X, Q = data.load_embeddings()                  # the full baseline workload: 200 queries x 20,000 docs
    if config.QUICK:                               # smoke test only (same as exp1's QUICK baseline)
        X, Q = X[:2_000], Q[:20]
    corpus = X.astype(np.float64).tolist()
    queries = Q.astype(np.float64).tolist()
    if args.cprofile:
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        res = pr.runcall(naive.search_naive, queries, corpus, 10)
        print(f"profiled_s {time.perf_counter() - t0!r}")
        pr.dump_stats(args.cprofile)
    else:
        res = naive.search_naive(queries, corpus, 10)
    print("top-3 of query 0:", res[0][:3])
