"""Stage 2C (Phase 2C) - the cosine-similarity matrix in pure Python, NumPy, Numba, Numba parallel
(+ fastmath) (report section 3.3)."""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from .. import bench, config, data, kernels
from .. import plotting as P

STAGE = "2c_vectorisation"

VERSIONS = {          # label -> (colour key, function, warm-up runs)
    "Pure Python": ("python", kernels.cosine_matrix_python, 1),
    "NumPy (BLAS)": ("numpy", kernels.cosine_matrix_numpy, 3),
    "Numba @njit": ("numba", kernels.cosine_matrix_numba, 3),
    "Numba parallel": ("numba_parallel", kernels.cosine_matrix_numba_parallel, 3),
    "Numba parallel+fastmath": ("numba_fastmath", kernels.cosine_matrix_numba_fastmath, 3),
}


def load():
    return data.load_embeddings()                     # float32 corpus (20000 x 768), queries (200 x 768)


def compile_cost(Q, X):
    """First call of a fresh (uncached) Numba function = compilation + run."""
    firsts, seconds = [], []
    for _ in range(5):                                   # 5 fresh dispatchers = 5 independent compilations
        fresh = njit(kernels.cosine_matrix_numba.py_func)
        t0 = time.perf_counter(); fresh(Q[:5], X[:500]); t1 = time.perf_counter(); fresh(Q[:5], X[:500]); t2 = time.perf_counter()
        firsts.append(t1 - t0); seconds.append(t2 - t1)
    out = {"first_compile_in_process_s": firsts[0],          # includes Numba / LLVM initialising itself
           "compile_median_s": float(np.median(firsts)), "second_call_s": float(np.median(seconds))}
    bench.update_summary(STAGE, numba_compile=out)
    return out


def cache_load_cost():
    """First call in a FRESH process when the compiled code is already in Numba's on-disk cache (cache=True)."""
    import subprocess
    import sys
    code = ("import time, numpy as np; from hpss import data, kernels; X, Q = data.load_embeddings(n=500); "
            "t0 = time.perf_counter(); kernels.cosine_matrix_numba(Q[:5], X); t1 = time.perf_counter(); "
            "kernels.cosine_matrix_numba(Q[:5], X); t2 = time.perf_counter(); print(t1 - t0, t2 - t1)")
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
            for _ in range(5)]                           # 5 fresh processes, median
    res = {"first_call_from_cache_s": float(np.median([float(o[0]) for o in runs])),
           "second_call_s": float(np.median([float(o[1]) for o in runs]))}
    bench.update_summary(STAGE, numba_cache=res)
    return res


# ---------------------------------------------------------------- memory layout & NumPy idioms (week 2)
@njit(cache=True)
def _row_norms_loop(A):
    n, d = A.shape
    out = np.empty(n)
    for i in range(n):
        s = 0.0
        for t in range(d):
            s += A[i, t] * A[i, t]
        out[i] = np.sqrt(s)
    return out


def layout_facts(X):
    """shape / dtype / strides / flags of the corpus and of typical views and copies."""
    idx = np.arange(0, len(X), 7)
    cases = {"X (C order)": X, "X.T (transpose)": X.T, "np.asfortranarray(X)": np.asfortranarray(X),
             "X[:, :128] (slice)": X[:, :128], "X[::2] (step slice)": X[::2], "X[idx] (fancy index)": X[idx]}
    rows = [{"array": k, "shape": str(a.shape), "strides": str(a.strides), "C_contiguous": a.flags["C_CONTIGUOUS"],
             "F_contiguous": a.flags["F_CONTIGUOUS"], "shares_memory_with_X": bool(np.shares_memory(a, X)),
             "MB": a.nbytes / 2**20} for k, a in cases.items()]
    return pd.DataFrame(rows)


def layout_experiments(Q, X):
    """Pairs of (good, bad) ways to do the same thing; warm-up + 5 runs each."""
    Xf = np.asfortranarray(X)
    Qn = data.normalize_rows(Q).astype(np.float32)
    idx = np.random.default_rng(0).choice(len(X), 2_000, replace=False)
    X64, Q64 = X.astype(np.float64), Qn.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    cand = np.random.default_rng(1).integers(0, len(X), (len(Qn), 100))
    Xc = X[cand]                                                   # (200, 100, 768) gathered candidates
    from ..kernels import candidate_scores

    def inplace():
        Y = X.copy()                                               # same copy in both versions (fair)
        Y /= norms
        return Y

    def outofplace():
        Y = X.copy()
        return Y / norms

    pairs = [
        ("row loop (Numba)", "C order (rows contiguous)", lambda: _row_norms_loop(X),
         "F order (rows strided)", lambda: _row_norms_loop(Xf)),
        ("GEMM Q @ X.T", "contiguous X[:, :384] copy", (lambda Xs=np.ascontiguousarray(X[:, :384]): Qn[:, :384] @ Xs.T),
         "strided view X[:, ::2]", lambda: Qn[:, ::2] @ X[:, ::2].T),
        ("select 2,000 rows", "slice X[:2000] (view)", lambda: X[:2000],
         "fancy index X[idx] (copy)", lambda: X[idx]),
        ("normalise rows", "in place: Y /= norms", inplace,
         "new array: Y / norms", outofplace),
        ("GEMM 200 x 20K x 768", "float32", lambda: Qn @ X.T,
         "float64", lambda: Q64 @ X64.T),
        ("scores Q . X", "Q @ X.T (BLAS)", lambda: Qn @ X.T,
         "einsum('qd,nd->qn') no optimize", lambda: np.einsum("qd,nd->qn", Qn, X)),
        ("re-rank 200 x 100 cands", "Numba, reads rows in place", lambda: candidate_scores(Qn, X, cand),
         "einsum('qd,qcd->qc') on gathered copy", lambda: np.einsum("qd,qcd->qc", Qn, X[cand])),
    ]
    rows = []
    for op, good_name, good, bad_name, bad in pairs:
        tg = bench.benchmark(good, warmup=2, runs=5)["median"]
        tb = bench.benchmark(bad, warmup=2, runs=5)["median"]
        rows.append({"operation": op, "good": good_name, "good_ms": tg * 1e3, "bad": bad_name, "bad_ms": tb * 1e3,
                     "ratio": tb / tg})
    df = pd.DataFrame(rows)
    peak_in, _ = bench.peak_memory_mb(inplace)
    peak_out, _ = bench.peak_memory_mb(outofplace)
    bench.save_csv(df, "memory_layout", STAGE)
    bench.update_summary(STAGE, memory_layout=df.to_dict("records"),
                         normalise_peak_mb={"in_place": peak_in, "new_array": peak_out},
                         gathered_candidates_mb=Xc.nbytes / 2**20)
    return df, {"in_place": peak_in, "new_array": peak_out}


def plot_layout(df):
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(df) + 1.4))
    y = np.arange(len(df))
    ax.barh(y - 0.19, df.good_ms, height=0.36, color=P.color("numpy"), edgecolor="white", label="good / layout-aware")
    ax.barh(y + 0.19, df.bad_ms, height=0.36, color=P.color("python"), edgecolor="white", label="bad / layout-unaware")
    for i, r in df.iterrows():
        ax.annotate(f"{r.ratio:,.1f}x", xy=(max(r.good_ms, r.bad_ms), i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=P.INK_2)
    ax.set_yticks(y, df.operation)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xscale("log"); ax.set_xlabel("time (ms, log); label = bad / good")
    ax.set_title("(d) Same result, different memory layout or idiom", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    P.savefig(fig, "fig4d_memory_layout", STAGE)
    r = df.set_index("operation").ratio
    bench.save_caption(STAGE, "fig4d_memory_layout", (
        f"Figure 4d: How the data is laid out and touched matters as much as the arithmetic: a fancy-index copy is "
        f"{r['select 2,000 rows']:,.0f}x slower than a slice view, an einsum that bypasses BLAS {r['scores Q . X']:.0f}x slower "
        f"than Q @ X.T, gathering candidate rows before re-ranking {r['re-rank 200 x 100 cands']:.0f}x slower than reading them in "
        f"place, and float64 {r['GEMM 200 x 20K x 768']:.1f}x slower than float32; row-major vs column-major access costs only "
        f"{r['row loop (Numba)']:.1f}x here because the 59 MB corpus fits in the 96 MB L3 cache."))
    return fig


def check_correctness(Q, X):
    ref = kernels.cosine_matrix_python(Q[:7], X[:300])
    return {name: float(np.abs(fn(Q[:7], X[:300]) - ref).max())
            for name, (_, fn, _) in VERSIONS.items() if name != "Pure Python"}


def main_comparison(Q, X, n_queries=20, n_docs=2_000):
    """All five versions on the same size (small enough for pure Python)."""
    if config.QUICK:
        n_docs = 300
    Qm, Xm = Q[:n_queries], X[:n_docs]
    rows = []
    for label, (key, fn, wu) in VERSIONS.items():
        r = bench.benchmark(fn, Qm, Xm, warmup=wu, runs=5)
        rows.append({"version": label, "key": key, "median_s": r["median"], "std_s": r["std"]})
        print(f"{label:>24}: {bench.fmt_time(r['median'])}")
    df = pd.DataFrame(rows)
    df["speedup_vs_python"] = df.median_s.iloc[0] / df.median_s
    df["gflops"] = 2 * n_queries * n_docs * X.shape[1] / df.median_s / 1e9
    bench.save_csv(df, "main_comparison", STAGE)
    bench.update_summary(STAGE, main=df.to_dict("records"), main_size={"queries": n_queries, "docs": n_docs})
    return df


def _numpy_1t(Qa, Xa):
    with bench.blas_threads(1):
        return kernels.cosine_matrix_numpy(Qa, Xa)


SWEEP = {"numpy": kernels.cosine_matrix_numpy, "numpy_1thread": _numpy_1t, "numba": kernels.cosine_matrix_numba,
         "numba_parallel": kernels.cosine_matrix_numba_parallel, "numba_fastmath": kernels.cosine_matrix_numba_fastmath}


def size_sweep(Q, X):
    """Batch (200 queries, GEMM) and single query (GEMV) for growing corpus sizes, no pure Python."""
    sizes = [10, 1_000] if config.QUICK else [10, 100, 1_000, 5_000, 20_000]
    rows = []
    for nq in [200, 1]:
        for nd in sizes:
            for key, fn in SWEEP.items():
                if key == "numba" and nq * nd > 200 * 5_000:
                    continue                              # serial Numba too slow at the largest size
                r = bench.benchmark(fn, Q[:nq], X[:nd], warmup=3, runs=5)
                rows.append({"n_queries": nq, "n_docs": nd, "method": key, "median_s": r["median"]})
    sweep = pd.DataFrame(rows)
    bench.save_csv(sweep, "size_sweep", STAGE)
    return sweep


def plot_figure4(main, sweep):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    sp = main.iloc[1:]
    P.bars(ax, [v.replace(" ", "\n", 1) for v in sp.version], sp.speedup_vs_python.tolist(), sp.key.tolist(),
           fmt="{:,.0f}x", log=True)
    ax.set_ylabel("speed-up over pure Python (log)")
    best = main.loc[main.speedup_vs_python.idxmax()]
    ax.set_title(f"(a) {best.version}: {best.speedup_vs_python:,.0f}x", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=8.5)
    names = {"numpy": "NumPy (BLAS, all threads)", "numpy_1thread": "NumPy (BLAS, 1 thread)", "numba": "Numba @njit",
             "numba_parallel": "Numba parallel", "numba_fastmath": "Numba parallel+fastmath"}
    for ax, nq, ttl in [(axes[1], 200, "(b) 200 queries (GEMM)"), (axes[2], 1, "(c) 1 query (GEMV)")]:
        g0 = sweep[sweep.n_queries == nq]
        for key in ["numpy", "numpy_1thread", "numba", "numba_parallel", "numba_fastmath"]:
            g = g0[g0.method == key]
            P.line(ax, g.n_docs, g.median_s * 1e3, "numpy" if key == "numpy_1thread" else key, names[key],
                   linestyle="--" if key == "numpy_1thread" else "-",
                   markerfacecolor="white" if key == "numpy_1thread" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("documents N"); ax.set_ylabel("time (ms, log)")
        ax.set_title(ttl, fontsize=10.5)
    axes[2].legend(fontsize=8, loc="upper left")
    P.savefig(fig, "fig4_vectorisation", STAGE)
    return fig


def finalize(main, sweep):
    m = main.set_index("key")
    nd = sweep.n_docs.max()
    g200 = sweep[(sweep.n_queries == 200) & (sweep.n_docs == nd)].set_index("method").median_s
    g1 = sweep[(sweep.n_queries == 1) & (sweep.n_docs == nd)].set_index("method").median_s
    bench.update_summary(STAGE, batch_200_largest_ms=(g200 * 1e3).to_dict(), single_query_largest_ms=(g1 * 1e3).to_dict())
    faster = "Numba fastmath" if g200["numba_fastmath"] < g200["numpy"] else "NumPy/BLAS"
    cap = (f"Figure 4: Compiling the same loops with Numba gives {m.loc['numba', 'speedup_vs_python']:.0f}x over pure Python, "
           f"prange {m.loc['numba_parallel', 'speedup_vs_python']:,.0f}x and prange+fastmath (SIMD) "
           f"{m.loc['numba_fastmath', 'speedup_vs_python']:,.0f}x, versus {m.loc['numpy', 'speedup_vs_python']:,.0f}x for "
           f"NumPy/BLAS. NumPy loses whenever its single-threaded normalisation temporaries dominate: for one query over "
           f"{nd:,} docs it needs {g1['numpy'] * 1e3:.1f} ms versus {g1['numba_fastmath'] * 1e3:.2f} ms for the fused Numba "
           f"kernel; for a 200-query batch {faster} is faster (Numba {g200['numba_fastmath'] * 1e3:.1f} ms vs NumPy "
           f"{g200['numpy'] * 1e3:.1f} ms).")
    bench.save_caption(STAGE, "fig4_vectorisation", cap)
    return cap


def main():
    P.setup_style()
    X, Q = load()
    print(compile_cost(Q, X))
    print(cache_load_cost())
    print(check_correctness(Q, X))
    print(layout_facts(X))
    layout, _ = layout_experiments(Q, X)
    plot_layout(layout)
    main_df = main_comparison(Q, X)
    sweep = size_sweep(Q, X)
    plot_figure4(main_df, sweep)
    print(finalize(main_df, sweep))


if __name__ == "__main__":
    main()
