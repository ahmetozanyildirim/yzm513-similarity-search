"""Independent, same-machine validation on the committed text corpus.

This is a separate TF-IDF + SVD experiment, not a reproduction of the archived
Windows MPNet timings. Every comparison uses identical vectors and 3 warmups /
5 timed runs. Source files and previous results are never overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from hpss import bench, config, data, kernels, naive, parallel
from hpss.search import search_numpy, topk_from_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 20000])
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if (args.queries < 1 or args.dimensions < 1 or args.features <= args.dimensions
            or not args.sizes or min(args.sizes) < 10):
        parser.error("positive queries/dimensions and sizes >=10 are required; features must exceed dimensions")
    if not args.smoke and (not 50 <= args.queries <= 200 or max(args.sizes) < 5000):
        parser.error("reported runs require 50-200 queries and >=5000 documents; use --smoke for tests")
    if args.output.exists():
        parser.error("output directory already exists; choose a fresh path")
    corpus, queries = data.load_corpus(), data.load_queries()
    if max(args.sizes) > len(corpus) or args.queries > len(queries):
        parser.error("requested workload exceeds the committed inputs")
    if args.dimensions >= min(max(args.sizes), args.features):
        parser.error("dimensions must be smaller than the fit matrix dimensions")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=args.features, ngram_range=(1, 2), min_df=1,
                                stop_words="english", sublinear_tf=True, dtype=np.float64)
    sparse_corpus = vectorizer.fit_transform(corpus.abstract.iloc[:max(args.sizes)])
    sparse_queries = vectorizer.transform(queries.text.iloc[:args.queries])
    reducer = TruncatedSVD(n_components=args.dimensions, n_iter=5, random_state=42)
    x = np.ascontiguousarray(data.normalize_rows(reducer.fit_transform(sparse_corpus)), dtype=np.float64)
    q = np.ascontiguousarray(data.normalize_rows(reducer.transform(sparse_queries)), dtype=np.float64)
    preparation_seconds = time.perf_counter() - started
    worker_count = min(4, os.cpu_count() or 1, numba.config.NUMBA_NUM_THREADS)
    if worker_count < 1:
        raise RuntimeError("no usable worker count")
    code_files = ["src/hpss/bench.py", "src/hpss/data.py", "src/hpss/search.py",
                  "src/hpss/kernels.py", "src/hpss/svd.py", "scripts/validate_independent.py"]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    summary = {
        "run_kind": "smoke" if args.smoke else "independent_experiment",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip(),
        "representation": {"name": "TF-IDF + TruncatedSVD; not MPNet", "dimensions": args.dimensions,
                           "vocabulary_size": sparse_corpus.shape[1], "max_features": args.features,
                           "fit_documents": len(x), "fit_queries": 0, "seed": 42},
        "environment": {**bench.system_info(), "machine": platform.machine()},
        "fingerprints": {"corpus_jsonl_sha256": sha(config.CORPUS_JSONL),
                         "queries_jsonl_sha256": sha(config.QUERIES_JSONL),
                         "code_sha256": {name: sha(config.ROOT / name) for name in code_files}},
        "preparation_seconds_excluded_from_search": preparation_seconds,
        "warmup": 3, "runs": 5, "queries": args.queries, "top_k": 10,
        "sizes": args.sizes, "workers": worker_count, "rows": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        flat = [{key: value for key, value in row.items() if key != "raw_times"} for row in summary["rows"]]
        pd.DataFrame(flat).to_csv(args.output / "timings.csv", index=False)

    for n in args.sizes:
        xn = x[:n]
        q_list, x_list = q.tolist(), xn.tolist()
        print(f"workload: {args.queries} queries x {n} documents x {args.dimensions} dimensions", flush=True)
        ref_timing = bench.benchmark(naive.search_naive, q_list, x_list, 10, warmup=3, runs=5,
                                     progress=f"naive-{n}")
        ref_idx, ref_scores = bench.to_arrays(ref_timing["result"])
        np.savez(args.output / f"reference_{n}.npz", idx=ref_idx, scores=ref_scores)
        baseline_seconds = ref_timing["median"]

        def record(name, result, output, workers):
            check = bench.compare_topk(*output, ref_idx, ref_scores)
            if not check["all_ok"]:
                raise AssertionError(f"{name} at N={n} failed exact comparison: {check}")
            row = {"n_documents": n, "n_queries": args.queries, "dimensions": args.dimensions,
                   "version": name, "workers": workers, "median_seconds": result["median"],
                   "mean_seconds": result["mean"], "std_seconds": result["std"],
                   "speedup_same_machine_same_vectors": baseline_seconds / result["median"],
                   "warmup": result["warmup"], "runs": result["runs"],
                   "raw_times": result["raw_times"], **check}
            summary["rows"].append(row)
            save()
            print(f"{name}: median={result['median']:.6f}s exact={check['all_ok']}", flush=True)

        record("python_naive", ref_timing, (ref_idx, ref_scores), 1)
        for threads in (1, worker_count):
            with bench.blas_threads(threads):
                timing = bench.benchmark(search_numpy, q, xn, 10, warmup=3, runs=5)
            record(f"numpy_blas_{threads}", timing, timing["result"], threads)
        numba.set_num_threads(worker_count)
        for name, func in (("numba_serial_matrix", kernels.cosine_matrix_numba),
                           ("numba_parallel_matrix", kernels.cosine_matrix_numba_parallel)):
            def execute(fn=func):
                return topk_from_scores(fn(q, xn), 10)
            timing = bench.benchmark(execute, warmup=3, runs=5)
            record(name, timing, timing["result"], 1 if name == "numba_serial_matrix" else worker_count)
        for threads in (1, worker_count):
            numba.set_num_threads(threads)
            timing = bench.benchmark(kernels.search_numba_fused, q, xn, 10, warmup=3, runs=5)
            record(f"numba_fused_{threads}", timing, timing["result"], threads)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            timing = bench.benchmark(parallel.search_threadpool, q, xn, 10, worker_count,
                                     executor=executor, warmup=3, runs=5)
        record(f"threadpool_{worker_count}", timing, timing["result"], worker_count)

    summary["all_exact_checks_passed"] = all(row["all_ok"] for row in summary["rows"])
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    frame = pd.DataFrame(summary["rows"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, part in frame.groupby("version", sort=False):
        ax.plot(part.n_documents, part.median_seconds * 1000, marker="o", label=name)
    ax.set(xlabel="Documents", ylabel="Median search time (ms)", yscale="log",
           title=f"Independent TF-IDF/SVD search: {args.queries} queries, {args.dimensions} dimensions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "scaling.png", dpi=150)
    fig.savefig(args.output / "scaling.pdf")
    plt.close(fig)
    print(f"completed {len(summary['rows'])} measured comparisons; {args.output}", flush=True)
    return summary


if __name__ == "__main__":
    main()
