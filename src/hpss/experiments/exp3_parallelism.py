"""Stage 3 (Phase 3) - parallelism and concurrency strategies (report section 4).

Every experiment writes its own CSV into results/3_parallelism/, so the parts can be run
separately; `scaling_figure()` and `summary_table()` read them back.

  A  GIL: pure-Python search on threads vs processes (+ pickling cost, cold start)
  B  BLAS threads (implicit parallelism) + batching (GEMV loop vs one GEMM)
  C  Numba prange (fused similarity + top-k)
  D  ThreadPoolExecutor + NumPy (with / without BLAS oversubscription)
  E  ProcessPoolExecutor + shared memory
  F  asyncio (simulated embedding-API latency)
  G  Dask (sharded scatter-gather; dask.array)
  H  Ray (tasks with ray.put, stateful actors)
"""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import bench, config, data, naive
from .. import parallel as par
from .. import plotting as P
from ..search import search_numpy_per_query

STAGE = "3_parallelism"
K = config.TOP_K
W_LIST = [1, 2, 4] if config.QUICK else [1, 2, 4, 8, 12, 16]


def load():
    X, Q = data.load_embeddings()
    Xn = np.ascontiguousarray(data.normalize_rows(X), dtype=np.float32)
    Qn = np.ascontiguousarray(data.normalize_rows(Q), dtype=np.float32)
    ref = np.load(config.baseline_topk_path())
    return X, Q, Xn, Qn, (ref["idx_20000"], ref["sc_20000"])


def correct(out, ref, n=None):
    r = ref if n is None else (ref[0][:n], ref[1][:n])
    return bench.compare_topk(*out, *r)["all_ok"]


def naive_full_seconds():
    """Naive baseline for the full workload (measured per-query time of stage 1 x 200)."""
    return bench.require_stage("1_baseline")["baseline"]["full_workload_s_estimate"]


def phase2_reference(Qn, Xn, ref):
    """Best Phase-2 version without parallelism: NumPy batch search on 1 BLAS thread."""
    r = bench.benchmark(par.search_blas, Qn, Xn, K, 1)
    out = {"median_s": r["median"], "correct": correct(r["result"], ref)}
    bench.update_summary(STAGE, phase2_reference=out)
    return out


# ---------------------------------------------------------------- A: GIL
def gil_experiment(X, Q, n_queries=48, n_docs=1_000):
    q_py = Q[:n_queries].astype(np.float64).tolist()
    c_py = X[:n_docs].astype(np.float64).tolist()
    ref = bench.to_arrays(naive.search_naive(q_py, c_py, K))
    t_seq = bench.benchmark(naive.search_naive, q_py, c_py, K, warmup=1, runs=5)["median"]
    rows = []
    for w in W_LIST:
        with ThreadPoolExecutor(w) as ex:
            r = bench.benchmark(par.search_threadpool_python, q_py, c_py, K, w, executor=ex, warmup=1, runs=5)
        rows.append({"strategy": "ThreadPool (pure Python)", "workers": w, "median_s": r["median"],
                     "correct": bench.compare_topk(*bench.to_arrays(r["result"]), *ref)["all_ok"]})
        with par.warm_pool(lambda: par.make_python_pool(c_py, w), w) as pool:
            r = bench.benchmark(par.search_processpool_python, q_py, K, w, pool, warmup=1, runs=5)
        rows.append({"strategy": "ProcessPool (pure Python)", "workers": w, "median_s": r["median"],
                     "correct": bench.compare_topk(*bench.to_arrays(r["result"]), *ref)["all_ok"]})
        print(f"w={w:>2}: thread {rows[-2]['median_s']:.2f} s | process {rows[-1]['median_s']:.2f} s")
    gil = pd.DataFrame(rows)
    gil["speedup"] = t_seq / gil.median_s
    bench.save_csv(gil, "A_gil_python", STAGE)
    bench.update_summary(STAGE, gil_sequential_s=t_seq, gil_workload={"queries": n_queries, "docs": n_docs})
    return gil, t_seq


def pickling_experiment(X, Q, workers=8, n_queries=48, n_docs=1_000):
    """Bad (corpus pickled with every task) vs good (initializer) vs cold start (pool creation included)."""
    q_py = Q[:n_queries].astype(np.float64).tolist()
    c_py = X[:n_docs].astype(np.float64).tolist()
    workers = min(workers, bench.n_cores())
    corpus_bytes = len(pickle.dumps(c_py))
    with par.warm_pool(lambda: par.make_python_pool(c_py, workers), workers) as pool:
        t_bad = bench.benchmark(par.search_processpool_bad, q_py, c_py, K, pool, warmup=1, runs=5)["median"]
        t_good = bench.benchmark(par.search_processpool_python, q_py, K, workers, pool, warmup=1, runs=5)["median"]
    cold = []
    for _ in range(5):
        t0 = time.perf_counter()
        with par.make_python_pool(c_py, workers) as pool:
            par.search_processpool_python(q_py, K, workers, pool)
        cold.append(time.perf_counter() - t0)
    out = {"workers": workers, "corpus_pickle_mb": corpus_bytes / 2**20, "bad_s": t_bad, "good_warm_s": t_good,
           "good_cold_s": float(np.median(cold)), "bad_total_pickled_mb": corpus_bytes * n_queries / 2**20}
    bench.update_summary(STAGE, pickling=out)
    return out


# ---------------------------------------------------------------- B: BLAS threads + batching
def blas_scaling(Qn, Xn, ref):
    rows = []
    for w in W_LIST:
        r = bench.benchmark(par.search_blas, Qn, Xn, K, w)
        rows.append({"strategy": "BLAS threads (NumPy batch)", "workers": w, "median_s": r["median"],
                     "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "B_blas_scaling", STAGE)
    return df


def _per_query(Qa, Xa, k, threads):
    with bench.blas_threads(threads):
        return search_numpy_per_query(Qa, Xa, k)


def batching_experiment(Qn, Xn, blas_df):
    """200 matrix-vector products (one per query) vs one matrix-matrix product."""
    out = {"gemv_loop_1thread_ms": bench.benchmark(_per_query, Qn, Xn, K, 1)["median"] * 1e3,
           "gemv_loop_allthreads_ms": bench.benchmark(_per_query, Qn, Xn, K, None)["median"] * 1e3,
           "gemm_batch_1thread_ms": blas_df.median_s.iloc[0] * 1e3,
           "gemm_batch_best_ms": blas_df.median_s.min() * 1e3}
    bench.update_summary(STAGE, batching=out)
    return out


# ---------------------------------------------------------------- C: Numba prange
def numba_scaling(Qn, Xn, ref):
    rows = []
    for w in W_LIST:
        r = bench.benchmark(par.search_numba_threads, Qn, Xn, K, w)
        rows.append({"strategy": "Numba prange (fused)", "workers": w, "median_s": r["median"],
                     "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "C_numba_scaling", STAGE)
    return df


# ---------------------------------------------------------------- D: ThreadPool + NumPy
def threadpool_scaling(Qn, Xn, ref):
    rows = []
    for w in W_LIST:
        with ThreadPoolExecutor(w) as ex:
            r = bench.benchmark(par.search_threadpool, Qn, Xn, K, w, 1, ex)
            r_os = bench.benchmark(par.search_threadpool, Qn, Xn, K, w, None, ex)
        rows.append({"strategy": "ThreadPool + NumPy (1 BLAS thread/worker)", "workers": w,
                     "median_s": r["median"], "correct": correct(r["result"], ref)})
        rows.append({"strategy": "ThreadPool + NumPy (oversubscribed)", "workers": w,
                     "median_s": r_os["median"], "correct": correct(r_os["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "D_threadpool_scaling", STAGE)
    return df


# ---------------------------------------------------------------- E: ProcessPool + shared memory
def processpool_shm_scaling(Qn, Xn, ref):
    rows = []
    for w in W_LIST:
        with par.SharedCorpusPool(Xn, w) as pool:
            r = bench.benchmark(pool.search, Qn, K)
        rows.append({"strategy": "ProcessPool + shared memory", "workers": w, "median_s": r["median"],
                     "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "E_processpool_shm_scaling", STAGE)
    return df


# ---------------------------------------------------------------- F: asyncio
def asyncio_experiment(Qn, Xn, ref, n_queries=100, latencies=(0.0, 0.02, 0.05)):
    Qa = Qn[:n_queries]
    rows = []
    variants = [("serial", par.serve_serial, {}), ("concurrent", par.serve_concurrent, {}),
                ("bounded(10)", par.serve_concurrent, {"max_in_flight": 10}), ("batched", par.serve_batched, {})]
    for lat in latencies:
        for name, fn, kw in variants:
            ts, out = [], None
            for i in range(6):                                    # 1 warm-up + 5 timed
                t, out = par.run_async(fn, Qa, Xn, K, lat, **kw)
                if i:
                    ts.append(t)
            if isinstance(out, list):
                out = (np.vstack([o[0] for o in out]), np.vstack([o[1] for o in out]))
            rows.append({"latency_s": lat, "variant": name, "median_s": float(np.median(ts)),
                         "correct": correct(out, ref, n_queries)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "F_asyncio", STAGE)
    return df


def sleep_resolution(target=0.02, n=20):
    """How long does asyncio.sleep(target) really take on this OS?"""
    import asyncio

    async def one():
        t0 = time.perf_counter()
        await asyncio.sleep(target)
        return time.perf_counter() - t0

    async def many():
        return [await one() for _ in range(n)]

    _, out = par.run_async(many)
    res = {"requested_ms": target * 1e3, "median_ms": float(np.median(out) * 1e3)}
    bench.update_summary(STAGE, asyncio_sleep_resolution=res)
    return res


# ---------------------------------------------------------------- G: Dask
def dask_experiment(Qn, Xn, ref):
    rows = []
    for s in W_LIST if config.QUICK else [1, 2, 4, 8, 16]:
        r = bench.benchmark(par.search_dask, Qn, Xn, K, s)
        rows.append({"strategy": "Dask delayed shards (threads)", "workers": s, "median_s": r["median"],
                     "correct": correct(r["result"], ref)})
    for rows_per_chunk in [20_000, 5_000, 2_500]:
        r = bench.benchmark(par.search_dask_array, Qn, Xn, K, rows_per_chunk)
        rows.append({"strategy": f"dask.array matmul (chunk={rows_per_chunk})", "workers": np.nan,
                     "median_s": r["median"], "correct": correct(r["result"], ref)})
    r = bench.benchmark(par.search_dask, Qn, Xn, K, 8, "processes", warmup=1, runs=5)
    rows.append({"strategy": "Dask delayed shards (processes)", "workers": 8, "median_s": r["median"],
                 "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "G_dask", STAGE)
    return df


# ---------------------------------------------------------------- H: Ray
def ray_experiment(Qn, Xn, ref, n_shards=8):
    try:
        t0 = time.perf_counter()
        rs = par.RaySearch(Xn, n_shards=n_shards)
        init_s = time.perf_counter() - t0
        r_t = bench.benchmark(rs.search_tasks, Qn, K)
        r_a = bench.benchmark(rs.search_actors, Qn, K)
        rows = [{"strategy": "Ray tasks (ray.put shards)", "workers": n_shards, "median_s": r_t["median"],
                 "correct": correct(r_t["result"], ref)},
                {"strategy": "Ray actors (stateful shards)", "workers": n_shards, "median_s": r_a["median"],
                 "correct": correct(r_a["result"], ref)}]
        rs.shutdown()
    except Exception as e:                       # Ray can fail on some Windows set-ups; the rest still runs
        print("Ray could not run:", repr(e)[:300])
        init_s, rows = None, []
    df = pd.DataFrame(rows, columns=["strategy", "workers", "median_s", "correct"])
    bench.save_csv(df, "H_ray", STAGE)
    bench.update_summary(STAGE, ray_init_s=init_s)
    return df, init_s


# ---------------------------------------------------------------- Figure 5 + summary
SCALING_FILES = {"B_blas_scaling": "BLAS threads (NumPy batch)", "C_numba_scaling": "Numba prange (fused)",
                 "D_threadpool_scaling": "ThreadPool + NumPy (1 BLAS thread/worker)",
                 "E_processpool_shm_scaling": "ProcessPool + shared memory", "G_dask": "Dask delayed shards (threads)"}
KEYS = {"BLAS threads (NumPy batch)": "numpy", "Numba prange (fused)": "numba_parallel",
        "ThreadPool + NumPy (1 BLAS thread/worker)": "threadpool", "ProcessPool + shared memory": "processpool",
        "Dask delayed shards (threads)": "dask"}


def _read(name):
    return pd.read_csv(config.stage_dir(STAGE) / f"{name}.csv")


def load_scaling():
    parts = []
    for f, strat in SCALING_FILES.items():
        df = _read(f)
        parts.append(df[df.strategy == strat])
    scal = pd.concat(parts, ignore_index=True)
    scal["speedup"] = scal.groupby("strategy").median_s.transform(lambda s: s.iloc[0] / s)
    gil = _read("A_gil_python")
    fits = {name: P.fit_amdahl(g.workers, g.speedup)
            for name, g in list(scal.groupby("strategy")) + list(gil.groupby("strategy"))}
    bench.save_csv(scal, "scaling_all", STAGE)
    bench.update_summary(STAGE, amdahl_p=fits)
    return scal, gil, fits


def scaling_figure(scal, gil, fits):
    cores = bench.n_cores()
    wmax = int(max(scal.workers.max(), gil.workers.max()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    wx = np.linspace(1, wmax, 100)
    for ax in axes:
        ax.plot(wx, wx, color=P.AXIS, linewidth=1, label="ideal (linear)")
        ax.axvline(cores, color=P.MUTED, linewidth=0.8)
        ax.annotate(f"{cores} physical cores", xy=(cores, wmax * 0.94), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=P.INK_2)
    ax = axes[0]
    for name, key in [("ProcessPool (pure Python)", "processpool"), ("ThreadPool (pure Python)", "threadpool")]:
        g = gil[gil.strategy == name]
        P.line(ax, g.workers, g.speedup, key, f"{name}  (Amdahl p={fits[name]:.2f})")
        ax.plot(wx, P.amdahl(wx, fits[name]), color=P.color(key), linewidth=1, alpha=0.45)
    gp = gil[gil.strategy.str.startswith("Process")].speedup.max()
    gt = gil[gil.strategy.str.startswith("Thread")].speedup.max()
    ax.set_xlabel("workers"); ax.set_ylabel("speed-up vs 1 worker")
    ax.set_title(f"(a) Pure Python: processes {gp:.1f}x, threads {gt:.1f}x (GIL)", fontsize=10.5)
    ax.set_xticks(sorted(gil.workers.unique())); ax.set_ylim(0, wmax + 0.5); ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    for name, key in KEYS.items():
        g = scal[scal.strategy == name]
        P.line(ax, g.workers, g.speedup, key, f"{name.split(' (')[0]}  (p={fits[name]:.2f})")
    ax.set_xlabel("workers / threads / shards"); ax.set_ylabel("speed-up vs 1 worker")
    ax.set_title("(b) Vectorised search: a serial top-k and shared bandwidth limit the scaling", fontsize=10.5)
    ax.set_xticks(sorted(scal.workers.unique())); ax.set_ylim(0, wmax + 0.5); ax.legend(fontsize=8.5, loc="upper left")
    P.savefig(fig, "fig5_parallel_scaling", STAGE)
    return fig


def summary_table():
    """Best setting of every strategy vs the naive baseline and vs the best Phase-2 version."""
    res = bench.load_summary(STAGE)
    t_p2 = res["phase2_reference"]["median_s"]
    naive_s = naive_full_seconds()
    rows = [{"strategy": "Phase-2 best: NumPy batch (1 BLAS thread)", "key": "numpy", "config": "1 thread",
             "median_s": t_p2, "correct": res["phase2_reference"]["correct"]}]
    for f, strat in SCALING_FILES.items():
        df = _read(f)
        df = df[df.strategy == strat]
        b = df.loc[df.median_s.idxmin()]
        rows.append({"strategy": strat, "key": KEYS[strat], "config": f"{int(b.workers)} workers",
                     "median_s": b.median_s, "correct": bool(df.correct.all())})
    for _, b in _read("H_ray").iterrows():
        rows.append({"strategy": b.strategy, "key": "ray", "config": f"{int(b.workers)} shards",
                     "median_s": b.median_s, "correct": bool(b.correct)})
    asy = _read("F_asyncio")
    a = asy[asy.latency_s == 0.02].set_index("variant").median_s
    rows.append({"strategy": "asyncio batched (20 ms API latency, 100 q)", "key": "asyncio", "config": "gather + 1 GEMM",
                 "median_s": a["batched"], "correct": bool(asy.correct.all())})
    s = pd.DataFrame(rows)
    s["speedup_vs_naive"] = naive_s / s.median_s
    s["speedup_vs_phase2"] = t_p2 / s.median_s
    bench.save_csv(s, "summary", STAGE)
    bench.update_summary(STAGE, summary=s.to_dict("records"), naive_full_s=naive_s)
    return s


def finalize(scal, gil, fits):
    gp = gil[gil.strategy.str.startswith("Process")].speedup.max()
    gt = gil[gil.strategy.str.startswith("Thread")].speedup.max()
    cap = (f"Figure 5: With pure Python, processes reach {gp:.1f}x (Amdahl p={fits['ProcessPool (pure Python)']:.2f}) while "
           f"threads stay at {gt:.1f}x because of the GIL. For the vectorised search, BLAS threads stop at "
           f"{scal[scal.strategy.str.startswith('BLAS')].speedup.max():.1f}x (p={fits['BLAS threads (NumPy batch)']:.2f}) "
           f"because the top-k selection after the matmul is single-threaded, whereas strategies that split the work into "
           f"query chunks or shards also parallelise the selection (ProcessPool+shared memory "
           f"p={fits['ProcessPool + shared memory']:.2f}) and the fused Numba kernel has no serial part at all "
           f"(p={fits['Numba prange (fused)']:.2f}).")
    bench.save_caption(STAGE, "fig5_parallel_scaling", cap)
    return cap


def main():
    P.setup_style()
    X, Q, Xn, Qn, ref = load()
    print(phase2_reference(Qn, Xn, ref))
    gil, _ = gil_experiment(X, Q)
    print(pickling_experiment(X, Q))
    blas = blas_scaling(Qn, Xn, ref)
    print(batching_experiment(Qn, Xn, blas))
    numba_scaling(Qn, Xn, ref)
    threadpool_scaling(Qn, Xn, ref)
    processpool_shm_scaling(Qn, Xn, ref)
    asyncio_experiment(Qn, Xn, ref)
    print(sleep_resolution())
    dask_experiment(Qn, Xn, ref)
    ray_experiment(Qn, Xn, ref)
    scal, gil, fits = load_scaling()
    scaling_figure(scal, gil, fits)
    print(summary_table())
    print(finalize(scal, gil, fits))


if __name__ == "__main__":
    main()
