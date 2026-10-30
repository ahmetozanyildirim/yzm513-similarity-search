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


def workload(corpus_list, query_list):
    """The benchmark workload: all 200 queries; corpus sizes 1K, 5K, 10K, 20K (spec). QUICK = smoke test only."""
    if config.QUICK:
        return query_list[:20], [1_000, 2_000]
    return query_list, config.CORPUS_SIZES


def naive_full(corpus_list, query_list):
    """The baseline exactly as the spec asks - nothing extrapolated: the naive search of ALL 200 queries against
    1K, 5K, 10K and 20K documents, 1 warm-up + 5 timed runs at every size. The 20K row is the baseline time.
    The top-k lists these runs return are stored as the correctness reference of every later stage."""
    queries, sizes = workload(corpus_list, query_list)
    rows, store = [], {}
    for n in sizes:
        r = bench.benchmark(naive.search_naive, queries, corpus_list[:n], K, warmup=1, runs=5,
                            progress=f"{len(queries)} queries x {n:,} docs")
        store[f"idx_{n}"], store[f"sc_{n}"] = bench.to_arrays(r["result"])
        rows.append({"n_docs": n, "queries": len(queries), "median_s": r["median"], "mean_s": r["mean"],
                     "std_s": r["std"], "min_s": r["min"], "max_s": r["max"],
                     "per_query_s": r["median"] / len(queries), "per_pair_us": r["median"] / len(queries) / n * 1e6,
                     "raw_times": r["raw_times"]})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "naive_scaling", STAGE)
    baseline = rows[-1]
    bench.update_summary(STAGE, baseline=baseline, naive_by_n=df.set_index("n_docs").median_s.to_dict())
    if config.QUICK:                          # smoke test: later stages still need the 200 x 20K reference
        compute_reference(np.asarray(corpus_list), query_list)              # writes the file
    else:
        np.savez(config.baseline_topk_path(), **store)
    return df, baseline, store


def _short_name(fn):
    """'<method 'sort' of 'list' objects>' -> 'list.sort', '<built-in method math.sqrt>' -> 'math.sqrt'"""
    if fn == "<lambda>":
        return "<lambda> (sort key)"
    m = re.match(r"<method '(\w+)' of '(\w+)' objects>", fn)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    m = re.match(r"<built-in method (?:builtins\.)?([\w.]+)>", fn)
    return m.group(1) if m else fn


def run_cprofile(corpus_list, query_list, baseline):
    """cProfile of search_naive only (runcall) on the full baseline workload (200 queries x 20K docs);
    its overhead is measured against the unprofiled baseline median of the same workload."""
    queries, sizes = workload(corpus_list, query_list)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.runcall(naive.search_naive, queries, corpus_list[:sizes[-1]], K)
    t_prof = time.perf_counter() - t0
    t_plain = baseline["median_s"]
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


def run_line_profiler(corpus_list, query_list, n_queries=10):
    """line_profiler slows this code down ~14x, so it runs on 10 queries against the FULL 20K corpus (~2 min).
    It reports shares of time per line - every query runs exactly the same lines, so the shares do not depend
    on the number of queries; no time is extrapolated from it."""
    from line_profiler import LineProfiler
    n_docs = workload(corpus_list, query_list)[1][-1]
    if config.QUICK:
        n_queries = 2
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


def run_pyspy(rate=200):
    """Sampling profiler: flame graph of scripts/profile_naive.py (the full baseline workload, ~3 min).

    --subprocesses is needed on Windows: the venv's python.exe is a launcher that starts
    the real interpreter as a child process. Returns the SVG path, or None if py-spy is unavailable.
    """
    import shutil
    import subprocess
    exe = shutil.which("py-spy") or str(Path(sys.executable).with_name("py-spy.exe"))
    out = config.stage_dir(STAGE) / "flamegraph.svg"
    try:
        subprocess.run([exe, "record", "--subprocesses", "--rate", str(rate), "--format", "flamegraph",
                        "-o", str(out), "--", sys.executable, str(config.ROOT / "scripts" / "profile_naive.py")],
                       check=True, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        print("py-spy could not run:", repr(e)[:200])
        return None
    return out


def reprofile_after_fix(corpus_list, query_list, n_queries=10):
    """'Profile first. Fix the top bottleneck. Re-profile.' (week 1): line profile of the version with
    norms computed once - what is the next bottleneck? Same 10 queries x full corpus as run_line_profiler."""
    from line_profiler import LineProfiler
    n_docs = workload(corpus_list, query_list)[1][-1]
    if config.QUICK:
        n_queries = 2
    normed = naive.normalize_python(corpus_list[:n_docs])
    lp = LineProfiler(naive.search_python_prenorm)
    lp.runcall(naive.search_python_prenorm, query_list[:n_queries], normed, K)
    stats = lp.get_stats()
    rows = []
    for (fname, _, func), entries in stats.timings.items():
        tot = sum(t for _, _, t in entries)
        for lineno, hits, t in entries:
            rows.append({"line": lineno, "hits": hits, "pct_of_function": 100 * t / tot,
                         "code": linecache.getline(fname, lineno).strip()})
    df = pd.DataFrame(rows).sort_values("pct_of_function", ascending=False).reset_index(drop=True)
    bench.save_csv(df, "line_profiler_after_fix", STAGE)
    bench.update_summary(STAGE, reprofile_top_lines=df.head(3)[["code", "pct_of_function"]].to_dict("records"))
    return df


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


def python_fixes(corpus_list, query_list, X, Q, baseline, store):
    """Test the bottleneck hypotheses: remove each one while staying in pure Python. Same full workload as the
    baseline (200 queries x 20K docs), 1 warm-up + 5 runs each; the naive row IS the baseline measurement."""
    queries, sizes = workload(corpus_list, query_list)
    n_docs, n_queries = sizes[-1], len(queries)
    c_normed = naive.normalize_python(corpus_list[:n_docs])  # done once at indexing time (not timed)
    X64 = data.normalize_rows(X[:n_docs].astype(np.float64))
    Q64 = Q[:n_queries].astype(np.float64)

    def numpy_1thread(Qm, Xn, k):
        with bench.blas_threads(1):                          # fair: NumPy on one core too
            return search_numpy(data.normalize_rows(Qm), Xn, k)

    ladder = {
        "fix 2: norms once": (naive.search_python_prenorm, (queries, c_normed, K)),
        "fix 1+3: sum(map) + heapq": (naive.search_python_builtin, (queries, c_normed, K)),
        "NumPy (1 thread, reference)": (numpy_1thread, (Q64, X64, K)),
    }
    ref = store[f"idx_{n_docs}"][:n_queries], store[f"sc_{n_docs}"][:n_queries]
    rows = [{"version": "naive (baseline)", "median_s": baseline["median_s"], "std_s": baseline["std_s"],
             "correct": True}]                                       # measured above, not re-run
    for name, (fn, args) in ladder.items():
        r = bench.benchmark(fn, *args, warmup=1, runs=5, progress=name if fn is not numpy_1thread else None)
        out = r["result"]
        idx, sc = bench.to_arrays(out) if isinstance(out, list) else out
        rows.append({"version": name, "median_s": r["median"], "std_s": r["std"],
                     "correct": bench.compare_topk(idx, sc, *ref)["all_ok"]})
    df = pd.DataFrame(rows)
    df["speedup"] = df.median_s.iloc[0] / df.median_s
    df["workload"] = f"{n_queries} queries x {n_docs:,} docs"
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
    """QUICK (smoke test) only: the timed runs used a subset, but later stages need the 200 x 20K reference.
    Same search_naive code, split over worker processes. In a real run the reference is the output of naive_full."""
    workers = workers or bench.n_cores()
    sizes = [1_000, 20_000]
    store, t0 = {}, time.perf_counter()
    with parallel.make_python_pool(X, workers) as pool:
        for n in sizes:
            out = parallel.search_processpool_python(query_list, K, workers, pool, n_docs=n, chunks_per_worker=4)
            store[f"idx_{n}"], store[f"sc_{n}"] = bench.to_arrays(out)
            print(f"N={n:>6,}: done ({time.perf_counter() - t0:.0f} s)")
    np.savez(config.baseline_topk_path(), **store)
    return store


def check_reference(corpus_list, query_list):
    """Spot check of the saved file: a fresh naive run of 3 queries x 1,000 docs gives exactly the stored results."""
    saved = np.load(config.baseline_topk_path())
    seq = bench.to_arrays(naive.search_naive(query_list[:3], corpus_list[:1000], K))
    return {"sizes": sorted(int(k[4:]) for k in saved.files if k.startswith("idx_")),
            "queries": int(saved["idx_1000"].shape[0]),
            "identical": bool(np.array_equal(seq[0], saved["idx_1000"][:3]) and np.array_equal(seq[1], saved["sc_1000"][:3]))}


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
    return (f"{cap}\n\nBaseline (measured, median of 5 runs): {baseline['queries']} queries x {baseline['n_docs']:,} docs "
            f"in {baseline['median_s']:.1f} s (std {baseline['std_s']:.2f} s) = {baseline['per_query_s'] * 1e3:.0f} ms per query, "
            f"{baseline['per_pair_us']:.1f} us per pair. Pure-Python fixes reach {fixes.speedup.iloc[2]:.1f}x; "
            f"NumPy (1 thread) {fixes.speedup.iloc[3]:,.0f}x.")


def main():
    P.setup_style()
    X, Q, corpus_list, query_list = load_lists()
    print(small_correctness(corpus_list, query_list, X, Q)[0])
    scaling, baseline, store = naive_full(corpus_list, query_list)
    print(scaling)
    prof, _, _, _ = run_cprofile(corpus_list, query_list, baseline)
    cos_lines, _, _ = run_line_profiler(corpus_list, query_list)
    plot_profiling(prof, cos_lines)
    print("flame graph:", run_pyspy())
    fixes = python_fixes(corpus_list, query_list, X, Q, baseline, store)
    print(fixes)
    print(reprofile_after_fix(corpus_list, query_list).head(5))
    print(memory_layout(corpus_list, X))
    print("saved reference:", check_reference(corpus_list, query_list))
    print(finalize(prof, cos_lines, fixes, baseline))


if __name__ == "__main__":
    main()
