"""Stage 1 (Phase 1) - naive baseline, profiling and the top-3 bottlenecks (report section 2)."""

import cProfile
import io
import linecache
import pstats
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import bench, config, data, naive, parallel
from .. import plotting as P
from ..search import search_numpy

STAGE = "1_baseline"
K = config.TOP_K


def load_lists():
    """Corpus and queries as lists of Python floats (conversion is not timed)."""
    X, Q = data.load_embeddings()
    return X, Q, X.astype(np.float64).tolist(), Q.astype(np.float64).tolist()


def small_correctness(corpus_list, query_list, X, Q):
    """Naive result vs float64 NumPy on 3 queries x 1,000 docs."""
    ref = naive.search_naive(query_list[:3], corpus_list[:1000], K)
    idx, sc = search_numpy(data.normalize_rows(Q[:3].astype(np.float64)),
                           data.normalize_rows(X[:1000].astype(np.float64)), K)
    return bench.compare_topk(idx, sc, *bench.to_arrays(ref)), ref


def baseline_time(corpus_list, query_list, n_queries=10):
    """Naive search of n_queries against all 20K docs; warm-up 1 + 5 timed runs."""
    if config.QUICK:
        corpus_list, n_queries = corpus_list[:2000], 2
    res = bench.benchmark(naive.search_naive, query_list[:n_queries], corpus_list, K, warmup=1, runs=5)
    per_query = res["median"] / n_queries
    out = {"n_docs": len(corpus_list), "queries_timed": n_queries, "median_s": res["median"],
           "mean_s": res["mean"], "std_s": res["std"], "per_query_s": per_query,
           "per_pair_us": per_query / len(corpus_list) * 1e6,
           "full_workload_s_estimate": per_query * len(query_list), "raw_times": res["raw_times"]}
    bench.update_summary(STAGE, baseline=out)
    return out


def naive_scaling(corpus_list, query_list, n_queries=5):
    rows = []
    sizes = [1_000, 2_000] if config.QUICK else config.CORPUS_SIZES
    for n in sizes:
        r = bench.benchmark(naive.search_naive, query_list[:n_queries], corpus_list[:n], K, warmup=1, runs=5)
        rows.append({"n_docs": n, "queries": n_queries, "median_s": r["median"], "std_s": r["std"],
                     "per_query_s": r["median"] / n_queries,
                     "per_pair_us": r["median"] / n_queries / n * 1e6})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "naive_scaling", STAGE)
    bench.update_summary(STAGE, naive_per_pair_us_by_n=df.set_index("n_docs").per_pair_us.to_dict())
    return df


def _short_name(fn):
    """'<method 'sort' of 'list' objects>' -> 'list.sort', '<built-in method math.sqrt>' -> 'math.sqrt'"""
    if fn == "<lambda>":
        return "<lambda> (sort key)"
    m = re.match(r"<method '(\w+)' of '(\w+)' objects>", fn)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    m = re.match(r"<built-in method (?:builtins\.)?([\w.]+)>", fn)
    return m.group(1) if m else fn


def run_cprofile(corpus_list, query_list, n_queries=5, n_docs=5_000):
    """cProfile of search_naive only (runcall), plus its overhead vs an unprofiled run."""
    if config.QUICK:
        n_docs = 1_000
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.runcall(naive.search_naive, query_list[:n_queries], corpus_list[:n_docs], K)
    t_prof = time.perf_counter() - t0
    t_plain = bench.benchmark(naive.search_naive, query_list[:n_queries], corpus_list[:n_docs], K,
                              warmup=0, runs=1)["median"]
    out_dir = config.stage_dir(STAGE)
    pr.dump_stats(out_dir / "cprofile.prof")
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(12)
    (out_dir / "cprofile.txt").write_text(buf.getvalue(), encoding="utf-8")
    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values())
    prof = pd.DataFrame([
        {"function": _short_name(fn), "file": Path(f).name, "ncalls": v[1], "tottime_s": v[2],
         "cumtime_s": v[3], "pct_tottime": 100 * v[2] / total}
        for (f, _, fn), v in st.stats.items()]).sort_values("tottime_s", ascending=False).reset_index(drop=True)
    bench.save_csv(prof, "cprofile_functions", STAGE)
    bench.update_summary(STAGE, cprofile={"profiled_s": t_prof, "plain_s": t_plain, "overhead_x": t_prof / t_plain})
    return prof, buf.getvalue(), t_prof, t_plain


def run_line_profiler(corpus_list, query_list, n_queries=2, n_docs=2_000):
    from line_profiler import LineProfiler
    lp = LineProfiler(naive.cosine_similarity_python, naive.search_naive)
    lp.runcall(naive.search_naive, query_list[:n_queries], corpus_list[:n_docs], K)
    buf = io.StringIO()
    lp.print_stats(stream=buf)
    (config.stage_dir(STAGE) / "line_profiler.txt").write_text(buf.getvalue(), encoding="utf-8")
    stats = lp.get_stats()
    rows = []
    for (fname, _, func), entries in stats.timings.items():
        tot = sum(t for _, _, t in entries)
        for lineno, hits, t in entries:
            rows.append({"function": func, "line": lineno, "hits": hits, "time_s": t * stats.unit,
                         "pct_of_function": 100 * t / tot, "code": linecache.getline(fname, lineno).strip()})
    lines = pd.DataFrame(rows)
    bench.save_csv(lines, "line_profiler_lines", STAGE)
    cos = lines[lines.function == "cosine_similarity_python"].sort_values("pct_of_function", ascending=False)
    srch = lines[lines.function == "search_naive"].sort_values("pct_of_function", ascending=False)
    return cos, srch, buf.getvalue()


def plot_profiling(prof, cos_lines):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    top = prof.head(6)
    ax = axes[0]
    P.bars(ax, [f"{r.function}\n{r.ncalls:,} calls" for r in top.itertuples()], top.pct_tottime.tolist(),
           ["python"] * len(top), fmt="{:.1f}%", horizontal=True)
    ax.set_xlabel("share of total time (tottime)")
    ax.set_title("(a) cProfile: almost all time is in the cosine function", fontsize=11)
    ax.set_xlim(0, 115)
    ax = axes[1]
    cl = cos_lines.head(6)
    labels = [c if len(c) < 38 else c[:36] + "..." for c in cl.code]
    P.bars(ax, labels, cl.pct_of_function.tolist(), ["python"] * len(cl), fmt="{:.1f}%", horizontal=True)
    ax.set_xlabel("share of time inside cosine_similarity_python")
    ax.set_title("(b) line_profiler: the loop and its 3 multiply-adds", fontsize=11)
    ax.set_xlim(0, cl.pct_of_function.max() * 1.25)
    ax.tick_params(axis="y", labelsize=9)
    P.savefig(fig, "fig1_profiling", STAGE)
    return fig


def python_fixes(corpus_list, query_list, X, Q, n_queries=10, n_docs=5_000):
    """Test the bottleneck hypotheses: remove each one while staying in pure Python."""
    if config.QUICK:
        n_docs = 1_000
    c_small = corpus_list[:n_docs]
    c_normed = naive.normalize_python(c_small)               # done once at indexing time (not timed)
    X64 = data.normalize_rows(X[:n_docs].astype(np.float64))
    Q64 = Q[:n_queries].astype(np.float64)

    def numpy_1thread(Qm, Xn, k):
        with bench.blas_threads(1):                          # fair: NumPy on one core too
            return search_numpy(data.normalize_rows(Qm), Xn, k)

    ladder = {
        "naive (baseline)": (naive.search_naive, (query_list[:n_queries], c_small, K)),
        "fix 2: norms once": (naive.search_python_prenorm, (query_list[:n_queries], c_normed, K)),
        "fix 1+3: sum(map) + heapq": (naive.search_python_builtin, (query_list[:n_queries], c_normed, K)),
        "NumPy (1 thread, reference)": (numpy_1thread, (Q64, X64, K)),
    }
    ref = bench.to_arrays(naive.search_naive(query_list[:n_queries], c_small, K))
    rows = []
    for name, (fn, args) in ladder.items():
        r = bench.benchmark(fn, *args, warmup=1, runs=5)
        out = r["result"]
        idx, sc = bench.to_arrays(out) if isinstance(out, list) else out
        rows.append({"version": name, "median_s": r["median"], "std_s": r["std"],
                     "correct": bench.compare_topk(idx, sc, *ref)["all_ok"]})
    df = pd.DataFrame(rows)
    df["speedup"] = df.median_s.iloc[0] / df.median_s
    bench.save_csv(df, "python_fixes", STAGE)
    bench.update_summary(STAGE, python_fixes=df.to_dict("records"))
    return df


def memory_layout(corpus_list, X):
    vec = corpus_list[0]
    py_bytes = sys.getsizeof(vec) + sum(sys.getsizeof(v) for v in vec)
    out = {"python_vector_kb": py_bytes / 1024, "numpy_vector_kb": X[0].nbytes / 1024,
           "ratio": py_bytes / X[0].nbytes, "python_corpus_mb": py_bytes * len(corpus_list) / 2**20,
           "numpy_corpus_mb": X.nbytes / 2**20}
    bench.update_summary(STAGE, memory_layout=out)
    return out


def compute_reference(X, query_list, workers=None):
    """Naive top-k for all 200 queries at every corpus size - the correctness reference of later stages.
    Same search_naive code, split over worker processes (otherwise ~5 min)."""
    workers = workers or bench.n_cores()
    sizes = [1_000, 20_000] if config.QUICK else config.CORPUS_SIZES       # later stages need the 20K reference
    store, t0 = {}, time.perf_counter()
    with parallel.make_python_pool(X, workers) as pool:
        for n in sizes:
            out = parallel.search_processpool_python(query_list, K, workers, pool, n_docs=n, chunks_per_worker=4)
            store[f"idx_{n}"], store[f"sc_{n}"] = bench.to_arrays(out)
            print(f"N={n:>6,}: done ({time.perf_counter() - t0:.0f} s)")
    np.savez(config.baseline_topk_path(), **store)
    return store


def check_reference(store, corpus_list, query_list):
    seq = bench.to_arrays(naive.search_naive(query_list[:3], corpus_list[:1000], K))
    return bool(np.array_equal(seq[0], store["idx_1000"][:3]) and np.array_equal(seq[1], store["sc_1000"][:3]))


def finalize(prof, cos_lines, fixes, baseline):
    cos_pct = float(prof.loc[prof.function == "cosine_similarity_python", "pct_tottime"].iloc[0])
    sort_pct = float(prof.loc[prof.function.str.contains("sort"), "pct_tottime"].sum())
    loop_pct = float(cos_lines.pct_of_function.head(4).sum())
    bench.update_summary(STAGE, cprofile_cosine_pct=cos_pct, cprofile_sort_pct=sort_pct, line_top4_pct=loop_pct)
    cap = (f"Figure 1: The naive search spends {cos_pct:.0f}% of its time inside cosine_similarity_python "
           f"(called once per query-document pair) and only {sort_pct:.1f}% sorting; inside the function, the loop header "
           f"and the three multiply-add lines take {loop_pct:.1f}% in near-equal shares - the cost is interpreter overhead "
           f"per float operation, and two of the three multiply-adds only recompute norms.")
    bench.save_caption(STAGE, "fig1_profiling", cap)
    return (f"{cap}\n\nBaseline: {baseline['per_query_s'] * 1e3:.0f} ms per query on {baseline['n_docs']:,} docs "
            f"({baseline['per_pair_us']:.1f} us per pair); 200 queries ~ {baseline['full_workload_s_estimate']:.0f} s. "
            f"Pure-Python fixes reach {fixes.speedup.iloc[2]:.1f}x; NumPy (1 thread) {fixes.speedup.iloc[3]:.0f}x.")


def main():
    P.setup_style()
    X, Q, corpus_list, query_list = load_lists()
    print(small_correctness(corpus_list, query_list, X, Q)[0])
    baseline = baseline_time(corpus_list, query_list)
    print(naive_scaling(corpus_list, query_list))
    prof, _, _, _ = run_cprofile(corpus_list, query_list)
    cos_lines, _, _ = run_line_profiler(corpus_list, query_list)
    plot_profiling(prof, cos_lines)
    fixes = python_fixes(corpus_list, query_list, X, Q)
    print(fixes)
    print(memory_layout(corpus_list, X))
    store = compute_reference(X, query_list)
    print("reference == sequential naive:", check_reference(store, corpus_list, query_list))
    print(finalize(prof, cos_lines, fixes, baseline))


if __name__ == "__main__":
    main()
