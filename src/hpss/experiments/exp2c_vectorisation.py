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
    fresh = njit(kernels.cosine_matrix_numba.py_func)
    t0 = time.perf_counter(); fresh(Q[:5], X[:500]); t1 = time.perf_counter(); fresh(Q[:5], X[:500]); t2 = time.perf_counter()
    out = {"first_call_s": t1 - t0, "second_call_s": t2 - t1}
    bench.update_summary(STAGE, numba_compile=out)
    return out


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
    print(check_correctness(Q, X))
    main_df = main_comparison(Q, X)
    sweep = size_sweep(Q, X)
    plot_figure4(main_df, sweep)
    print(finalize(main_df, sweep))


if __name__ == "__main__":
    main()
