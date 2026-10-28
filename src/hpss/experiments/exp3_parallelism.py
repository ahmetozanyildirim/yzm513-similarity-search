"""Stage 3 (Phase 3) - parallelism and concurrency strategies (report section 4).

Every experiment writes its own CSV into results/3_parallelism/, so the parts can be run
separately; `scaling_figure()` and `summary_table()` read them back.

  A  GIL: pure-Python search on threads vs processes (+ pickling cost, cold start)
  B  BLAS threads (implicit parallelism) + batching (GEMV loop vs one GEMM)
  C  Numba prange (fused similarity + top-k)
  D  ThreadPoolExecutor + NumPy (with / without BLAS oversubscription)
  E  ProcessPoolExecutor + shared memory
  F  asyncio (simulated embedding-API latency)
  G  Dask (sharded scatter-gather; dask.array; schedulers; tree vs flat merge; task timeline)
  H  Ray (tasks with ray.put, stateful actors, ray.wait, retries after failures, timeline)
  I  Parallel preprocessing: the DF/IDF word count with Dask Bag and Ray (week 5 example on our corpus)
Plus the week 3-5 concepts: granularity sweep, effective serial fraction f, race conditions and
lock granularity, free-threaded Python, producer-consumer micro-batching, memory-mapped corpus,
and the week-5 "why not the others?" checklist.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numba import njit, prange

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


def blas_serial_share(Qn, Xn, threads=(1, None)):
    """Measured split of the batched NumPy search into matmul (parallel in BLAS) and top-k
    selection (np.argpartition, single-threaded) - the serial fraction behind the BLAS curve."""
    from ..search import SearchEngine
    eng = SearchEngine().fit(Xn)
    rows = []
    for t in threads:
        with bench.blas_threads(t):
            for _ in range(3):
                eng.search(Qn, K)
            parts = []
            for _ in range(10):
                tm = {}
                eng.search(Qn, K, timings=tm)
                parts.append(tm)
        med = pd.DataFrame(parts).median()
        total = med.sum()
        rows.append({"blas_threads": t or "all", "matmul_ms": med["matmul"] * 1e3, "topk_ms": med["topk"] * 1e3,
                     "other_ms": med["normalise"] * 1e3, "serial_share": (med["topk"] + med["normalise"]) / total})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "B_blas_serial_share", STAGE)
    bench.update_summary(STAGE, blas_serial_share=df.to_dict("records"))
    return df


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


@njit(parallel=True, cache=True)
def score_histogram_racy(S, n_bins):
    """WRONG ON PURPOSE: every thread increments ONE shared histogram -> lost updates (a data race)."""
    hist = np.zeros(n_bins, dtype=np.int64)
    nq, n = S.shape
    for i in prange(nq):
        for j in range(n):
            b = min(max(int((S[i, j] + 1.0) * 0.5 * n_bins), 0), n_bins - 1)
            hist[b] += 1                                  # read-modify-write on shared memory
    return hist


@njit(parallel=True, cache=True)
def score_histogram_local(S, n_bins):
    """Correct: each query (thread) owns one row of `local`; combine once at the end."""
    nq, n = S.shape
    local = np.zeros((nq, n_bins), dtype=np.int64)
    for i in prange(nq):
        for j in range(n):
            b = min(max(int((S[i, j] + 1.0) * 0.5 * n_bins), 0), n_bins - 1)
            local[i, b] += 1                              # private to this iteration -> no race
    return local.sum(axis=0)


def race_demo(Qn, Xn, n_bins=50, repeats=5):
    """Histogram of all 200 x 20K similarity scores, computed in parallel two ways."""
    S = Qn @ Xn.T
    expected = S.size
    rows = []
    for name, fn in [("shared histogram (race)", score_histogram_racy), ("private rows, combine once", score_histogram_local)]:
        fn(S, n_bins)                                     # compile
        counts = [int(fn(S, n_bins).sum()) for _ in range(repeats)]
        t = bench.benchmark(fn, S, n_bins, warmup=1, runs=5)["median"]
        rows.append({"version": name, "expected_total": expected, "totals_in_5_runs": counts,
                     "lost_updates_max": expected - min(counts), "correct": all(c == expected for c in counts),
                     "ms": t * 1e3})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "C_race_demo", STAGE)
    bench.update_summary(STAGE, race_demo=df.to_dict("records"))
    return df


def find_free_threaded_python():
    """A free-threaded interpreter (python3.14t): $HPSS_FT_PYTHON or `uv python find 3.14t`."""
    exe = os.environ.get("HPSS_FT_PYTHON")
    if not exe and shutil.which("uv"):
        r = subprocess.run(["uv", "python", "find", "3.14t"], capture_output=True, text=True)
        exe = r.stdout.strip() if r.returncode == 0 else None
    return exe


def free_threading_experiment(X, Q, n_queries=48, n_docs=1_000):
    """Same pure-Python thread test on the normal (GIL) interpreter and on a free-threaded one."""
    subset = config.stage_dir(STAGE) / "_ft_subset.json"
    subset.write_text(json.dumps({"queries": Q[:n_queries].astype(float).tolist(),
                                  "corpus": X[:n_docs].astype(float).tolist()}))
    script = str(config.ROOT / "scripts" / "gil_free_threading.py")
    runs = {"GIL (standard)": sys.executable, "free-threaded (no GIL)": find_free_threaded_python()}
    rows, counters = [], []
    try:
        for label, exe in runs.items():
            if not exe:
                print(f"{label}: interpreter not found - install with `uv python install 3.14t`")
                continue
            out = json.loads(subprocess.run([exe, script, str(subset)], capture_output=True, text=True,
                                            check=True).stdout.strip().splitlines()[-1])
            base = out["search"]["1"]
            for w, t in out["search"].items():
                rows.append({"interpreter": label, "python": out["python"], "threads": int(w), "seconds": t,
                             "speedup": base / t})
            for name, c in out["counter"].items():
                counters.append({"interpreter": label, "version": name, **c, "lost_worst": c["expected"] - c["count"]})
    finally:
        subset.unlink(missing_ok=True)
    ft, cnt = pd.DataFrame(rows), pd.DataFrame(counters)
    bench.save_csv(ft, "A_free_threading", STAGE)
    bench.save_csv(cnt, "A_counter_race", STAGE)
    bench.update_summary(STAGE, switch_interval_s=sys.getswitchinterval())
    return ft, cnt


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


def processpool_memmap_scaling(Qn, ref):
    """Same as E, but each worker memory-maps the .npy file (np.load(..., mmap_mode='r'))."""
    rows = []
    for w in W_LIST:
        with par.MemmapCorpusPool(config.CORPUS_EMB, w) as pool:
            r = bench.benchmark(pool.search, Qn, K)
        rows.append({"strategy": "ProcessPool + memmap file", "workers": w, "median_s": r["median"],
                     "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "E_processpool_memmap_scaling", STAGE)
    return df


def granularity_sweep(Qn, Xn, ref, workers=8, sizes=(1, 2, 5, 10, 25, 50, 100, 200)):
    """'Granularity is the master dial' (week 5): queries per task from 1 to 200 with a fixed pool.
    S is measured against the single-threaded NumPy batch search; f = (1/S - 1/N) / (1 - 1/N)."""
    t_serial = bench.load_summary(STAGE)["phase2_reference"]["median_s"]
    workers = min(workers, os.cpu_count() or workers)
    rows = []
    with ThreadPoolExecutor(workers) as ex, par.SharedCorpusPool(Xn, workers) as pool:
        for g in sizes:
            for name, fn in [("ThreadPool + NumPy", lambda g=g: par.search_threadpool(Qn, Xn, K, workers, 1, ex, per_task=g)),
                             ("ProcessPool + shared memory", lambda g=g: pool.search(Qn, K, per_task=g))]:
                r = bench.benchmark(fn, warmup=2, runs=5)
                S = t_serial / r["median"]
                rows.append({"strategy": name, "queries_per_task": g, "tasks": -(-len(Qn) // g),
                             "median_s": r["median"], "speedup": S,
                             "f_eff": (1 / S - 1 / workers) / (1 - 1 / workers), "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "granularity", STAGE)
    bench.update_summary(STAGE, granularity_workers=workers)
    return df


def batch_cost(Qn, Xn, sizes=(1, 2, 4, 8, 25, 50, 200)):
    """One search call for m queries on 1 BLAS thread. m = 1 runs a GEMV; m >= 2 runs a GEMM, which first
    packs (copies) the corpus into cache-sized blocks - a fixed cost that only larger batches amortise."""
    rows = []
    with bench.blas_threads(1):
        for m in sizes:
            t = bench.benchmark(par.search_numpy, Qn[:m], Xn, K, warmup=3, runs=15)["median"]
            rows.append({"queries_in_call": m, "blas_routine": "GEMV" if m == 1 else "GEMM",
                         "ms": t * 1e3, "ms_per_query": t * 1e3 / m})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "batch_cost", STAGE)
    return df


def plot_granularity(df):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for name, key in [("ThreadPool + NumPy", "threadpool"), ("ProcessPool + shared memory", "processpool")]:
        g = df[df.strategy == name]
        P.line(axes[0], g.queries_per_task, g.median_s * 1e3, key, name)
        P.line(axes[1], g.queries_per_task, g.f_eff, key, name)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("queries per task (granularity, log)")
    axes[0].set_yscale("log"); axes[0].set_ylabel("time for 200 queries (ms, log)")
    best = df.loc[df.median_s.idxmin()]
    axes[0].set_title(f"(a) Sweet spot: {int(best.queries_per_task)} queries/task ({best.strategy.split(' +')[0]})", fontsize=10.5)
    axes[1].axhline(1, color=P.MUTED, linewidth=0.8)
    axes[1].set_ylabel("effective serial fraction f")
    axes[1].set_title("(b) f > 1: the overhead costs more than the whole job", fontsize=10.5)
    axes[1].legend(fontsize=8.5)
    P.savefig(fig, "fig5c_granularity", STAGE)
    g1 = df[df.queries_per_task == 1].median_s.min()
    g2 = df[df.queries_per_task == 2].median_s.min()
    whole = df[df.queries_per_task == df.queries_per_task.max()].median_s.min()
    bench.save_caption(STAGE, "fig5c_granularity", (
        f"Figure 5c: Granularity is the master dial: with {bench.load_summary(STAGE).get('granularity_workers', 8)} workers the "
        f"200 queries run fastest as tasks of {int(best.queries_per_task)} queries ({best.median_s * 1e3:.1f} ms, effective serial "
        f"fraction f = {best.f_eff:.2f}); one task with all queries leaves the other workers idle ({whole * 1e3:.1f} ms), and "
        f"tiny tasks drown in overhead - 2-query tasks ({g2 * 1e3:.1f} ms) are even slower than 1-query tasks ({g1 * 1e3:.1f} ms) "
        f"because BLAS switches from a matrix-vector to a matrix-matrix product that first packs the whole corpus."))
    return fig


def queue_experiment(Qn, Xn, ref, batches=(1, 8, 32, 200), delays=(0.0, 0.0005)):
    """Producer-consumer with dynamic batching: throughput and per-query latency."""
    rows = []
    for delay in delays:
        for b in batches:
            par.serve_queue(Qn, Xn, K, b, delay)                  # warm-up
            runs = [par.serve_queue(Qn, Xn, K, b, delay) for _ in range(5)]
            totals = [r[2] for r in runs]
            lat = np.concatenate([r[1] for r in runs])
            rows.append({"producer_delay_ms": delay * 1e3, "max_batch": b, "total_s": float(np.median(totals)),
                         "throughput_qps": len(Qn) / float(np.median(totals)),
                         "latency_p50_ms": float(np.percentile(lat, 50) * 1e3),
                         "latency_p95_ms": float(np.percentile(lat, 95) * 1e3),
                         "correct": correct(runs[-1][0], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "F_queue_microbatching", STAGE)
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
def _retry_broken_pool(fn, tries=2):
    """Dask's 'processes' scheduler spawns a fresh pool of 16 processes on every compute; on Windows a worker
    occasionally dies while starting (BrokenProcessPool). Retry once - visibly - instead of losing the notebook run."""
    from concurrent.futures.process import BrokenProcessPool
    for i in range(tries):
        try:
            return fn()
        except BrokenProcessPool as e:
            if i == tries - 1:
                raise
            print(f"process pool broke while starting ({e!r:.80}) - retrying once")


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
    r = _retry_broken_pool(lambda: bench.benchmark(par.search_dask, Qn, Xn, K, 8, "processes", warmup=1, runs=5))
    rows.append({"strategy": "Dask delayed shards (processes)", "workers": 8, "median_s": r["median"],
                 "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "G_dask", STAGE)
    return df


def dask_schedulers(Qn, Xn, ref, n_shards=8):
    """The same delayed graph on Dask's local schedulers and on a distributed LocalCluster."""
    rows = []
    for sched in ["synchronous", "threads", "processes"]:
        r = _retry_broken_pool(lambda: bench.benchmark(par.search_dask, Qn, Xn, K, n_shards, sched, warmup=1, runs=5))
        rows.append({"scheduler": sched, "median_s": r["median"], "startup_s": 0.0, "correct": correct(r["result"], ref)})
    try:
        t0 = time.perf_counter()
        dist = par.DaskDistributedSearch(Xn, n_workers=4, n_shards=n_shards)
        startup = time.perf_counter() - t0
        r = bench.benchmark(dist.search, Qn, K, warmup=2, runs=5)
        rows.append({"scheduler": "distributed (LocalCluster, shards scattered once)", "median_s": r["median"],
                     "startup_s": startup, "correct": correct(r["result"], ref)})
        dist.close()
    except Exception as e:                       # distributed can fail on some set-ups; the rest still runs
        print("dask.distributed could not run:", repr(e)[:200])
    df = pd.DataFrame(rows)
    bench.save_csv(df, "G_dask_schedulers", STAGE)
    return df


def dask_tree_vs_flat(Qn, Xn, ref, n_shards=64, fan_in=4):
    rows = []
    for name, fn, stats in [("flat merge (one narrow waist)", lambda: par.search_dask(Qn, Xn, K, n_shards),
                             par.dask_graph_stats(n_shards)),
                            (f"tree merge (fan-in {fan_in})", lambda: par.search_dask_tree(Qn, Xn, K, n_shards, fan_in),
                             par.dask_graph_stats(n_shards, fan_in))]:
        r = bench.benchmark(fn, warmup=1, runs=5)
        rows.append({"graph": name, **stats, "median_s": r["median"], "correct": correct(r["result"], ref)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "G_dask_tree_vs_flat", STAGE)
    return df


def dask_timeline(Qn, Xn, n_shards=8):
    """Task timeline of the threaded scheduler (dask.diagnostics.Profiler): what each worker thread did."""
    import dask
    from dask.diagnostics import Profiler
    spans = par._chunks(len(Xn), n_shards)
    parts = [dask.delayed(par._shard_topk)(Qn, Xn[a:b], a, K, dask_key_name=f"shard-{i}") for i, (a, b) in enumerate(spans)]
    merged = dask.delayed(par.merge_topk)(parts, K, dask_key_name="merge")
    with bench.blas_threads(1):
        merged.compute(scheduler="threads")                  # warm-up
        with Profiler() as prof:
            merged.compute(scheduler="threads")
    t0 = min(r.start_time for r in prof.results)
    rows = [{"task": str(r.key[0] if isinstance(r.key, tuple) else r.key).split("-")[0], "worker": r.worker_id,
             "start_ms": (r.start_time - t0) * 1e3, "end_ms": (r.end_time - t0) * 1e3} for r in prof.results]
    df = pd.DataFrame(rows)
    df = df[df.task.isin(["shard", "merge"])]
    workers = {w: i for i, w in enumerate(sorted(df.worker.unique()))}
    df["lane"] = df.worker.map(workers)
    bench.save_csv(df, "G_dask_timeline", STAGE)
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(workers) + 1.6))
    for _, r in df.iterrows():
        ax.barh(r.lane, r.end_ms - r.start_ms, left=r.start_ms, height=0.6, edgecolor="white",
                color=P.color("dask" if r.task == "shard" else "python"))
    ax.set_yticks(list(workers.values()), [f"thread {i}" for i in workers.values()])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (ms)")
    mer = df[df.task == "merge"]
    ax.set_title(f"Dask task timeline: {n_shards} shard tasks in parallel (green), one merge at the end "
                 f"(orange, {float((mer.end_ms - mer.start_ms).sum()):.2f} ms) = the narrow waist", fontsize=10.5)
    P.savefig(fig, "fig5d_dask_timeline", STAGE)
    sh = df[df.task == "shard"]
    wall = float(df.end_ms.max() - df.start_ms.min())
    merge_ms = float((mer.end_ms - mer.start_ms).sum())
    bench.save_caption(STAGE, "fig5d_dask_timeline", (
        f"Figure 5d: Dask's task stream for one sharded search: the {len(sh)} shard tasks run side by side on "
        f"{sh.worker.nunique()} threads ({float((sh.end_ms - sh.start_ms).median()):.1f} ms each) and the single merge - the "
        f"graph's narrow waist - takes {merge_ms:.2f} ms, only {100 * merge_ms / wall:.0f}% of the {wall:.1f} ms wall time, "
        f"so a flat merge is enough at this scale."))
    return df, fig


# ---------------------------------------------------------------- H: Ray
def ray_experiment(Qn, Xn, ref, n_shards=8):
    try:
        t0 = time.perf_counter()
        rs = par.RaySearch(Xn, n_shards=n_shards)
        init_s = time.perf_counter() - t0
        r_t = bench.benchmark(rs.search_tasks, Qn, K)
        r_a = bench.benchmark(rs.search_actors, Qn, K)
        r_w = bench.benchmark(rs.search_tasks_wait, Qn, K)
        r_f = bench.benchmark(rs.search_flaky, Qn, K, warmup=1, runs=5)
        rows = [{"strategy": "Ray tasks (ray.put shards)", "workers": n_shards, "median_s": r_t["median"],
                 "correct": correct(r_t["result"], ref)},
                {"strategy": "Ray actors (stateful shards)", "workers": n_shards, "median_s": r_a["median"],
                 "correct": correct(r_a["result"], ref)},
                {"strategy": "Ray tasks + ray.wait (merge as they arrive)", "workers": n_shards,
                 "median_s": r_w["median"], "correct": correct(r_w["result"], ref)},
                {"strategy": "Ray tasks, 30% of attempts fail, max_retries=10", "workers": n_shards,
                 "median_s": r_f["median"], "correct": correct(r_f["result"], ref)}]
        rs.timeline(config.stage_dir(STAGE) / "ray_timeline.json")
        rs.shutdown()
    except Exception as e:                       # Ray can fail on some Windows set-ups; the rest still runs
        print("Ray could not run:", repr(e)[:300])
        init_s, rows = None, []
    df = pd.DataFrame(rows, columns=["strategy", "workers", "median_s", "correct"])
    bench.save_csv(df, "H_ray", STAGE)
    bench.update_summary(STAGE, ray_init_s=init_s)
    return df, init_s


# ---------------------------------------------------------------- I: parallel preprocessing (DF / IDF)
def _word_set(doc):
    return set(doc.split())


def _df_chunk(docs):
    c = Counter()
    for d in docs:
        c.update(set(d.split()))
    return c


def _merge_counters(a, b):
    return a + b


def df_pipeline(partitions=(2, 8, 50, 500), ray_chunks=16):
    """Week 5's DF/IDF example on OUR 20K abstracts: in how many documents does each word occur?
    Pure-Python work (sets and a Counter) -> threads are blocked by the GIL, processes are needed.
    Serial Counter vs Dask Bag (threads / processes, partition sweep) vs Ray tasks with a tree merge."""
    import dask
    import dask.bag as db
    texts = [t.lower() for t in data.load_corpus()["abstract"]]
    ref = dict(_df_chunk(texts))
    rows = []

    def add(engine, parts, r):
        rows.append({"engine": engine, "partitions": parts, "median_s": r["median"], "correct": dict(r["result"]) == ref})

    add("serial Counter", 1, bench.benchmark(_df_chunk, texts, warmup=1, runs=5))

    def bag(n):
        return db.from_sequence(texts, npartitions=n).map(_word_set).flatten().frequencies()

    add("Dask Bag, threads (GIL)", 8, bench.benchmark(lambda: bag(8).compute(scheduler="threads"), warmup=1, runs=5))
    add("Dask Bag, processes (new pool every compute)", 8,       # every run is a cold start by design: Dask's default
        _retry_broken_pool(lambda: bench.benchmark(lambda: bag(8).compute(scheduler="processes"), warmup=1, runs=5)))
    with ProcessPoolExecutor(bench.n_cores()) as pool, dask.config.set(pool=pool):
        for n in partitions:
            add("Dask Bag, processes (pool reused)", n,
                bench.benchmark(lambda n=n: bag(n).compute(scheduler="processes"), warmup=1, runs=5))
    try:
        import ray
        t0 = time.perf_counter()
        ray.init(num_cpus=bench.n_cores(), include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
        init_s = time.perf_counter() - t0
        count, merge = ray.remote(_df_chunk), ray.remote(_merge_counters)
        refs = [ray.put(texts[a:b]) for a, b in par._chunks(len(texts), ray_chunks)]     # stored once

        def run():
            level = [count.remote(r) for r in refs]
            while len(level) > 1:                                                     # tree merge, depth log2
                level = [merge.remote(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
                         for i in range(0, len(level), 2)]
            return ray.get(level[0])

        add("Ray tasks + tree merge", ray_chunks, bench.benchmark(run, warmup=1, runs=5))
        rows[-1]["startup_s"] = init_s
        ray.shutdown()
    except Exception as e:
        print("Ray could not run:", repr(e)[:200])
    df = pd.DataFrame(rows)
    df["speedup_vs_serial"] = df.median_s.iloc[0] / df.median_s
    bench.save_csv(df, "I_df_pipeline", STAGE)
    bench.update_summary(STAGE, df_vocabulary=len(ref), df_words=sum(len(t.split()) for t in texts))
    return df


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
    scal["efficiency"] = scal.speedup / scal.workers                      # S / N
    n = scal.workers.where(scal.workers > 1)
    scal["f_eff"] = (1 / scal.speedup - 1 / n) / (1 - 1 / n)              # week 5 estimator (upper bound)
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
    mm = _read("E_processpool_memmap_scaling")
    b = mm.loc[mm.median_s.idxmin()]
    rows.append({"strategy": b.strategy, "key": "processpool", "config": f"{int(b.workers)} workers",
                 "median_s": b.median_s, "correct": bool(mm.correct.all())})
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


def week5_checklist():
    """The five questions of the week-5 'Project 1 Phase 3: Checklist', answered with our numbers."""
    import psutil
    res = bench.load_summary(STAGE)
    gran = _read("granularity")
    best = gran.loc[gran.median_s.idxmin()]
    share = pd.DataFrame(res["blas_serial_share"])
    corpus_mb = 20_000 * 768 * 4 / 2**20
    q_chunk_kb = best.queries_per_task * 768 * 4 / 1024
    rows = [
        ("1. Does the data fit in RAM?",
         f"yes: corpus {corpus_mb:.0f} MB vs {psutil.virtual_memory().total / 2**30:.0f} GB RAM (and it fits the 96 MB L3)",
         "Dask's out-of-core benefit does not apply"),
        ("2. Work per task vs overhead?",
         f"granularity sweep: best {int(best.queries_per_task)} queries/task ({best.median_s * 1e3:.1f} ms, f={best.f_eff:.2f}); "
         f"1 query/task gives f={gran[gran.queries_per_task == 1].f_eff.max():.2f}; a GEMM call has a fixed "
         f"packing cost of ~{_read('batch_cost').ms.iloc[1]:.1f} ms",
         "tasks must carry tens of queries"),
        ("3. How much data moves per task?",
         f"corpus stored once (shared memory / ray.put / scatter); per task only {q_chunk_kb:.0f} KB of queries out and "
         f"{best.queries_per_task * K * 12 / 1024:.1f} KB of results back",
         "move the query, not the data"),
        ("4. Is the serial part small?",
         f"no for BLAS-only: top-k is {share.serial_share.iloc[-1]:.0%} of the time with all BLAS threads; "
         "splitting by query chunks or shards parallelises it",
         "fixed in phase 4 with a parallel top-k"),
        ("5. Workers x BLAS threads <= cores?",
         "yes: pools use 1 BLAS thread per worker; the oversubscribed ThreadPool variant shows what happens otherwise",
         "D_threadpool_scaling.csv"),
    ]
    df = pd.DataFrame(rows, columns=["question", "our answer", "consequence"])
    bench.save_csv(df, "week5_checklist", STAGE)
    return df


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
    print(free_threading_experiment(X, Q))
    print(batching_experiment(Qn, Xn, blas))
    print(blas_serial_share(Qn, Xn))
    numba_scaling(Qn, Xn, ref)
    print(race_demo(Qn, Xn))
    threadpool_scaling(Qn, Xn, ref)
    processpool_shm_scaling(Qn, Xn, ref)
    processpool_memmap_scaling(Qn, ref)
    plot_granularity(granularity_sweep(Qn, Xn, ref))
    print(batch_cost(Qn, Xn))
    print(queue_experiment(Qn, Xn, ref))
    asyncio_experiment(Qn, Xn, ref)
    print(sleep_resolution())
    dask_experiment(Qn, Xn, ref)
    print(dask_schedulers(Qn, Xn, ref))
    print(dask_tree_vs_flat(Qn, Xn, ref))
    dask_timeline(Qn, Xn)
    ray_experiment(Qn, Xn, ref)
    print(df_pipeline())
    scal, gil, fits = load_scaling()
    scaling_figure(scal, gil, fits)
    print(summary_table())
    print(week5_checklist())
    print(finalize(scal, gil, fits))


if __name__ == "__main__":
    main()
