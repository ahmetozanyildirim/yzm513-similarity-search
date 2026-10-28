"""Phase 2C - the same operation implemented four (+1) ways.

Operation: the cosine-similarity matrix between queries Q (q x d) and corpus X (n x d)

        S[i, j] = (Q[i] . X[j]) / (|Q[i]| * |X[j]|)

This is exactly the hot loop of the search engine. All versions follow the same
algorithm (norms once per row, then one dot product per pair) so that the only
thing that changes is *how* the loops are executed.

  1. cosine_matrix_python            - interpreted Python loops
  2. cosine_matrix_numpy             - vectorised: two normalisations + one BLAS matmul
  3. cosine_matrix_numba             - the loops of (1), compiled by Numba (@njit)
  4. cosine_matrix_numba_parallel    - (3) with prange over documents (@njit(parallel=True))
  5. cosine_matrix_numba_fastmath    - (4) + fastmath=True so LLVM may reorder the
                                       float sum and use SIMD instructions
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


# 1 --------------------------------------------------------------------
def cosine_matrix_python(Q, X):
    Ql, Xl = Q.tolist(), X.tolist()
    d = len(Ql[0])
    qn = [math.sqrt(sum(v * v for v in q)) for q in Ql]
    xn = [math.sqrt(sum(v * v for v in x)) for x in Xl]
    S = [[0.0] * len(Xl) for _ in Ql]
    for i in range(len(Ql)):
        q = Ql[i]
        for j in range(len(Xl)):
            x = Xl[j]
            s = 0.0
            for t in range(d):
                s += q[t] * x[t]
            S[i][j] = s / (qn[i] * xn[j])
    return np.array(S)


# 2 --------------------------------------------------------------------
def cosine_matrix_numpy(Q, X):
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Qn @ Xn.T


# 3 --------------------------------------------------------------------
@njit(cache=True)
def cosine_matrix_numba(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    qn = np.empty(nq)
    for i in range(nq):
        s = 0.0
        for t in range(d):
            s += Q[i, t] * Q[i, t]
        qn[i] = math.sqrt(s)
    xn = np.empty(n)
    for j in range(n):
        s = 0.0
        for t in range(d):
            s += X[j, t] * X[j, t]
        xn[j] = math.sqrt(s)
    S = np.empty((nq, n))
    for i in range(nq):
        for j in range(n):
            s = 0.0
            for t in range(d):
                s += Q[i, t] * X[j, t]
            S[i, j] = s / (qn[i] * xn[j])
    return S


# 4 --------------------------------------------------------------------
@njit(parallel=True, cache=True)
def cosine_matrix_numba_parallel(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    qn = np.empty(nq)
    for i in prange(nq):
        s = 0.0
        for t in range(d):
            s += Q[i, t] * Q[i, t]
        qn[i] = math.sqrt(s)
    xn = np.empty(n)
    for j in prange(n):
        s = 0.0
        for t in range(d):
            s += X[j, t] * X[j, t]
        xn[j] = math.sqrt(s)
    S = np.empty((nq, n))
    for j in prange(n):              # documents are split across the threads
        for i in range(nq):
            s = 0.0
            for t in range(d):
                s += Q[i, t] * X[j, t]
            S[i, j] = s / (qn[i] * xn[j])
    return S


# 5 --------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def cosine_matrix_numba_fastmath(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    qn = np.empty(nq)
    for i in prange(nq):
        s = 0.0
        for t in range(d):
            s += Q[i, t] * Q[i, t]
        qn[i] = math.sqrt(s)
    xn = np.empty(n)
    for j in prange(n):
        s = 0.0
        for t in range(d):
            s += X[j, t] * X[j, t]
        xn[j] = math.sqrt(s)
    S = np.empty((nq, n))
    for j in prange(n):
        for i in range(nq):
            s = 0.0
            for t in range(d):
                s += Q[i, t] * X[j, t]
            S[i, j] = s / (qn[i] * xn[j])
    return S


# ======================================================================
# Parallel top-k selection and exact re-rank (Phase 4: removing the serial part)
# ======================================================================

@njit(parallel=True, cache=True)
def topk_rows(S, k):
    """Row-wise top-k of a score matrix, one thread per block of rows.

    np.argpartition does the same selection but on a single thread; after the
    BLAS matmul is parallel, that single-threaded selection becomes the serial
    fraction that limits the speed-up (Amdahl).
    """
    nq, n = S.shape
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_val = np.empty((nq, k), dtype=np.float64)
    for i in prange(nq):
        best_v = np.full(k, -np.inf)
        best_i = np.full(k, -1, dtype=np.int64)
        for j in range(n):
            s = S[i, j]
            if s > best_v[0]:
                p = 0
                while p + 1 < k and best_v[p + 1] < s:
                    best_v[p] = best_v[p + 1]
                    best_i[p] = best_i[p + 1]
                    p += 1
                best_v[p] = s
                best_i[p] = j
        for r in range(k):
            out_idx[i, r] = best_i[k - 1 - r]
            out_val[i, r] = best_v[k - 1 - r]
    return out_idx, out_val


@njit(parallel=True, fastmath=True, cache=True)
def candidate_scores(Qn, Xn, cand):
    """Exact dot products of each query with its own candidate rows of Xn.
    Reads the candidate rows in place (no (q x c x d) gather copy)."""
    nq, c = cand.shape
    d = Qn.shape[1]
    out = np.empty((nq, c), dtype=np.float64)
    for i in prange(nq):
        for r in range(c):
            j = cand[i, r]
            s = 0.0
            for t in range(d):
                s += Qn[i, t] * Xn[j, t]
            out[i, r] = s
    return out


# ======================================================================
# Fused "similarity + top-k" kernel (used in Phase 3 / 4)
# ======================================================================

@njit(parallel=True, fastmath=True, cache=True)
def search_numba_fused(Qn, Xn, k):
    """Top-k search on unit-length rows without ever building the q x n score matrix.

    Each query (prange -> one thread per query block) streams over the corpus and
    keeps its k best scores in a tiny sorted buffer (insertion sort, ascending).
    """
    nq, d = Qn.shape
    n = Xn.shape[0]
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_val = np.empty((nq, k), dtype=np.float64)
    for i in prange(nq):
        best_v = np.full(k, -np.inf)
        best_i = np.full(k, -1, dtype=np.int64)
        for j in range(n):
            s = 0.0
            for t in range(d):
                s += Qn[i, t] * Xn[j, t]
            if s > best_v[0]:                      # better than the current k-th best
                p = 0
                while p + 1 < k and best_v[p + 1] < s:
                    best_v[p] = best_v[p + 1]
                    best_i[p] = best_i[p + 1]
                    p += 1
                best_v[p] = s
                best_i[p] = j
        for r in range(k):                         # descending order
            out_idx[i, r] = best_i[k - 1 - r]
            out_val[i, r] = best_v[k - 1 - r]
    return out_idx, out_val
