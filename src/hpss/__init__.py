"""hpss - High-Performance Similarity Search (YZM 513, Project 1).

Modules by project phase:
    data, config, bench, plotting   shared infrastructure
    naive                           Phase 1  - pure-Python baseline
    sparse_dense, svd, kernels      Phase 2  - representations, dimensionality reduction, JIT
    parallel                        Phase 3  - parallelism & concurrency strategies
    search                          Phase 4  - the final SearchEngine
"""

import os as _os

# Two thread pools live in this process: OpenBLAS (NumPy matmul) and Numba's
# OpenMP pool (prange). After a prange loop, idle OpenMP threads keep spinning
# for a while and steal the cores that the next BLAS call needs - measured in
# stage 4 (experiments/exp4_final_system.thread_contention) (BLAS matmul up to 4x slower).
# PASSIVE makes idle OpenMP threads sleep instead. Must be set before Numba
# starts its pool, i.e. before the first parallel call - importing hpss first
# guarantees that. Set HPSS_NO_THREAD_FIX=1 to see the default behaviour.
if not _os.environ.get("HPSS_NO_THREAD_FIX"):
    _os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    _os.environ.setdefault("KMP_BLOCKTIME", "0")
