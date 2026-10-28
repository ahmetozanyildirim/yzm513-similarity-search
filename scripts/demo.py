"""Live demo of the final search engine (presentation: 30-second demo).

    python scripts/demo.py                                   # interactive prompt
    python scripts/demo.py "diffusion models for video"      # one query
    python scripts/demo.py --compare-naive "..."             # also run the naive baseline on the same query
"""

import argparse
import time

import numpy as np

from hpss import config, data, naive
from hpss.search import SearchEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--compare-naive", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    corpus = data.load_corpus()
    X, _ = data.load_embeddings()
    # configuration chosen in notebook 04 (falls back to SVD-128 + 100-candidate re-rank)
    from hpss import bench
    kw = bench.load_summary("4_final_system").get("final_config")
    kw = kw or {"n_components": 128, "n_candidates": 100, "topk": "numba"}
    engine = SearchEngine(**kw).fit(X)
    model = data.load_model()
    data.embed_texts(["warm-up"], model, show_progress=False)
    engine.search(X[:1], args.k)
    print(f"ready: {len(X):,} arXiv abstracts indexed in {time.perf_counter() - t0:.1f} s  {kw}\n")
    corpus_list = X.astype(np.float64).tolist() if args.compare_naive else None

    def run(text):
        t1 = time.perf_counter()
        q = data.embed_texts([text], model, show_progress=False)
        t2 = time.perf_counter()
        idx, sc = engine.search(q, args.k)
        t3 = time.perf_counter()
        print(f"  embedding: {1e3 * (t2 - t1):6.1f} ms   |   search: {1e3 * (t3 - t2):6.2f} ms")
        for r, (i, s) in enumerate(zip(idx[0], sc[0]), 1):
            d = corpus.iloc[i]
            print(f"  {r}. [{s:.3f}] {d.title[:95]}  ({d.primary_category}, {str(d.published)[:7]})")
        if corpus_list is not None:
            t4 = time.perf_counter()
            res = naive.search_naive(q.astype(np.float64).tolist(), corpus_list, args.k)
            t5 = time.perf_counter()
            same = [i for i, _ in res[0]] == list(idx[0])
            print(f"  naive baseline: {t5 - t4:.2f} s  ->  {(t5 - t4) / (t3 - t2):,.0f}x slower, "
                  f"same top-{args.k}: {same}")
        print()

    if args.query:
        run(" ".join(args.query))
        return
    while True:
        try:
            text = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in {"", "q", "quit", "exit"}:
            break
        run(text)


if __name__ == "__main__":
    main()
