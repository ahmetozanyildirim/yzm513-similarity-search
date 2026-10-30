"""Benchmarking, correctness and bookkeeping helpers (adapted from the course
starter `benchmark_utils.py`).

Rules enforced here (project spec, section 2):
  * every benchmark does warm-up runs first (Numba compiles on the first call)
  * at least 5 timed runs; we report median, mean and std
  * results are compared against the baseline with tolerance 1e-5
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ======================================================================
# Timing & memory
# ======================================================================

def benchmark(func, *args, warmup=config.WARMUP, runs=config.RUNS, progress=None, **kwargs):
    """Time `func(*args, **kwargs)` with `warmup` discarded calls and `runs` timed calls.

    Returns a dict with median/mean/std/min/max (seconds), the raw times and the
    result of the last call (so the caller can check correctness).
    progress: a label -> print every call's time as it finishes (for runs of minutes).
    """
    result = None
    for i in range(warmup):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        if progress:
            print(f"{progress}: warm-up {i + 1}/{warmup} {time.perf_counter() - t0:.1f} s", flush=True)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
        if progress:
            print(f"{progress}: run {i + 1}/{runs} {times[-1]:.1f} s", flush=True)
    t = np.asarray(times)
    return {
        "median": float(np.median(t)),
        "mean": float(t.mean()),
        "std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
        "runs": runs,
        "warmup": warmup,
        "raw_times": t.tolist(),
        "result": result,
    }


def peak_memory_mb(func, *args, **kwargs):
    """Run once under tracemalloc; return (peak MB, result).

    tracemalloc sees NumPy allocations (NumPy reports them to it) but not
    memory allocated privately inside BLAS/LAPACK, so treat it as a lower bound.
    """
    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2**20, result


_BLAS_CTRL = None


def blas_threads(n):
    """Context manager limiting BLAS (OpenBLAS/MKL) to n threads; None = no limit.

    threadpoolctl.threadpool_limits() re-scans every loaded library on each call
    (~4 ms on Windows) - far too slow to sit inside a timed function. We build the
    controller once and reuse it (~0.01 ms per call).
    """
    global _BLAS_CTRL
    if _BLAS_CTRL is None:
        from threadpoolctl import ThreadpoolController
        _BLAS_CTRL = ThreadpoolController()
    return _BLAS_CTRL.limit(limits=n, user_api="blas")


def fmt_time(seconds: float) -> str:
    """Human-readable time: 1.23 s / 4.56 ms / 7.89 us."""
    if seconds != seconds:          # NaN
        return "–"
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} us"


# ======================================================================
# Top-k result handling & correctness
# ======================================================================

def to_arrays(list_results):
    """Convert [[(doc_idx, score), ...], ...] (naive format) to (idx, scores) arrays."""
    idx = np.array([[i for i, _ in row] for row in list_results], dtype=np.int64)
    sc = np.array([[s for _, s in row] for row in list_results], dtype=np.float64)
    return idx, sc


def compare_topk(idx, scores, ref_idx, ref_scores, tol=config.TOL):
    """Check a top-k result against the reference (the naive baseline).

    Rank r is correct when its score matches the reference score at rank r
    within `tol`. Different indices are allowed only when the two documents
    are tied (their scores differ by < tol) - float32 vs float64 rounding can
    legitimately swap such near-ties.
    """
    idx = np.asarray(idx)
    scores = np.asarray(scores, dtype=np.float64)
    diff = np.abs(scores - ref_scores)
    score_ok = diff <= tol
    same_idx = idx == ref_idx
    return {
        "all_ok": bool(score_ok.all()),
        "max_abs_score_diff": float(diff.max()),
        "exact_index_match_rate": float(same_idx.mean()),
        "tie_swaps": int((~same_idx & score_ok).sum()),
        "violations": int((~score_ok).sum()),
    }


def recall_at_k(idx_approx, idx_exact):
    """Mean fraction of the exact top-k that the approximate top-k recovered."""
    hits = [len(set(a) & set(b)) / len(b) for a, b in zip(idx_approx, idx_exact)]
    return float(np.mean(hits))


# ======================================================================
# Environment description (goes into the report introduction)
# ======================================================================

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def system_info() -> dict:
    import numba
    import psutil
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    blas = [{k: d.get(k) for k in ("internal_api", "num_threads", "architecture", "version")}
            for d in threadpool_info() if d.get("user_api") == "blas"]
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "numba": numba.__version__,
        "blas": blas,
        "numba_threads": numba.config.NUMBA_NUM_THREADS,
        "in_colab": "google.colab" in sys.modules,
    }


# ======================================================================
# Saving results (raw data + key numbers + captions for the report)
# ======================================================================

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "result"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, name, stage) -> Path:
    path = config.stage_dir(stage) / (name if name.endswith(".json") else f"{name}.json")
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    return path


def save_csv(rows, name, stage) -> Path:
    """rows: list of dicts or a DataFrame -> results/<stage>/<name>.csv (raw benchmark data)"""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = config.stage_dir(stage) / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def _results_path(stage):
    return config.stage_dir(stage) / "results.json"


def update_summary(stage: str, **values) -> Path:
    """Merge key numbers into results/<stage>/results.json (the report is written from it)."""
    path = _results_path(stage)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(_jsonable(values))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_summary(stage: str) -> dict:
    """results/<stage>/results.json of an earlier stage ({} if it has not been run yet)."""
    path = _results_path(stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def require_stage(stage: str) -> dict:
    """Like load_summary, but explains which notebook/script to run first."""
    data = load_summary(stage)
    if not data:
        raise RuntimeError(f"results of stage {stage!r} not found in {config.RESULTS} - run that stage first "
                           f"(python -m hpss.experiments.exp{stage.split('_')[0]})")
    return data


def save_caption(stage: str, fig_id: str, caption: str) -> Path:
    """Store a conclusion-style figure caption in results/<stage>/results.json -> "captions"."""
    data = load_summary(stage)
    caps = data.get("captions", {})
    caps[fig_id] = caption
    return update_summary(stage, captions=caps)


def n_cores() -> int:
    """Physical cores (parallel speedup is bounded by these, not by SMT threads)."""
    import psutil
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1
