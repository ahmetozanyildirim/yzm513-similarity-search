"""Vectorised search building blocks + the final search engine (Phase 4).

Pipeline of `SearchEngine`:

    raw embeddings --(normalise once)--> X  (n x 768, float32, unit rows)
                   --(optional: project to k dims with SVD)--> Xr (n x k)

    query --> normalise --> [reduced] candidate search (BLAS matmul + argpartition)
          --> exact re-rank of the candidates in full 768-d --> top-k

The exact re-rank computes baseline-compatible scores for the selected candidates
(tolerance 1e-5). Candidate selection remains approximate: a true neighbour not
selected in reduced space cannot be recovered by re-ranking.
"""

from __future__ import annotations

import time

import numpy as np

from .data import normalize_rows


def topk_from_scores(S, k):
    """Row-wise top-k of a score matrix, sorted descending.

    argpartition is O(n) per row (a selection, not a sort); only the k winners
    are then sorted. A full argsort would be O(n log n).
    """
    S = np.atleast_2d(S)
    if (S.ndim != 2 or not S.shape[0] or not S.shape[1]
            or not np.isfinite(S).all()):
        raise ValueError("top-k scores must be a non-empty finite 2D matrix")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= S.shape[1]:
        raise ValueError("k must be an integer between 1 and the document count")
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(S, idx, axis=1)


def search_numpy(Qn, Xn, k):
    """Exact search on unit-length rows: one BLAS matmul + top-k selection."""
    return topk_from_scores(Qn @ Xn.T, k)


def search_numpy_per_query(Qn, Xn, k):
    """Same result, but one matrix-VECTOR product per query (how online serving
    sees the problem). Used to show why batching queries matters."""
    idx = np.empty((len(Qn), k), dtype=np.int64)
    val = np.empty((len(Qn), k), dtype=Xn.dtype)
    for i, q in enumerate(Qn):
        idx[i], val[i] = topk_from_scores(Xn @ q, k)
    return idx, val


class SearchEngine:
    """data -> embedding -> (dimensionality reduction) -> search -> top-k.

    Parameters
    ----------
    n_components : int or None   None = exact search in the full embedding space.
    reducer      : "truncated_svd" | "power_iteration" | "pca"
    n_candidates : int           candidates fetched in the reduced space and then
                                 re-ranked exactly (only used with n_components).
    topk         : "numpy"  -> np.argpartition (single thread) + gather/einsum re-rank
                   "numba"  -> parallel top-k selection + in-place re-rank (kernels.py)
    dtype        : np.float32 halves memory traffic vs float64 and is accurate to ~1e-7.
    """

    def __init__(self, n_components=None, reducer="truncated_svd", n_candidates=100,
                 topk="numpy", dtype=np.float32, seed=0):
        self.n_components = n_components
        self.reducer = reducer
        self.n_candidates = n_candidates
        self.topk = topk
        self.dtype = dtype
        self.seed = seed

    # ------------------------------------------------------------------
    def fit(self, X):
        from .svd import reduce
        self.X = np.ascontiguousarray(normalize_rows(np.asarray(X, dtype=np.float64)), dtype=self.dtype)
        self.n_docs = len(self.X)
        if self.n_components:
            fit = reduce(self.X.astype(np.float64), self.n_components, method=self.reducer,
                         seed=self.seed)
            self.components_ = fit["components"].astype(self.dtype)      # k x d
            self.mean_ = None if fit["mean"] is None else fit["mean"].astype(self.dtype)
            self.Xr = np.ascontiguousarray(normalize_rows(self._project(self.X)), dtype=self.dtype)
        return self

    def _project(self, A):
        if self.mean_ is not None:
            A = A - self.mean_
        return A @ self.components_.T

    # ------------------------------------------------------------------
    def search(self, Q, k=10, timings=None):
        """Return (idx, scores) - both (n_queries x k), scores are exact cosines.
        Pass a dict as `timings` to get the time of every stage (profiling)."""
        from .kernels import candidate_scores, topk_rows
        if not hasattr(self, "X"):
            raise ValueError("fit the search engine before searching")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= self.n_docs:
            raise ValueError("k must be an integer between 1 and the document count")
        clock = time.perf_counter
        t0 = clock()
        Qn = np.ascontiguousarray(normalize_rows(np.atleast_2d(np.asarray(Q, dtype=self.dtype))))
        select = topk_rows if self.topk == "numba" else topk_from_scores
        if not self.n_components:
            t1 = clock()
            S = Qn @ self.X.T                                        # BLAS GEMM
            t2 = clock()
            out = select(S, k)
            if timings is not None:
                timings.update(normalise=t1 - t0, matmul=t2 - t1, topk=clock() - t2)
            return out
        c = min(max(self.n_candidates, k), self.n_docs)
        Qr = np.ascontiguousarray(normalize_rows(self._project(Qn)))
        t1 = clock()
        S = Qr @ self.Xr.T                                           # BLAS GEMM in k dims
        t2 = clock()
        cand, _ = select(S, c)                                       # cheap, approximate
        t3 = clock()
        if self.topk == "numba":
            exact = candidate_scores(Qn, self.X, cand)               # exact cosines, no copy
        else:
            exact = np.einsum("qd,qcd->qc", Qn, self.X[cand])        # exact cosines of c candidates
        order, scores = topk_from_scores(exact, k)
        out = np.take_along_axis(cand, order, axis=1), scores
        if timings is not None:
            timings.update(normalise_project=t1 - t0, matmul=t2 - t1, topk_candidates=t3 - t2,
                           rerank=clock() - t3)
        return out

    def memory_mb(self):
        mb = self.X.nbytes / 2**20
        if self.n_components:
            mb += (self.Xr.nbytes + self.components_.nbytes) / 2**20
        return mb
