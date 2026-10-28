"""Phase 2A - sparse vs. dense representations, with our own dot-product kernels.

We compute the score matrix  S = Q @ X^T  (queries x documents) for three
storage combinations. The *math* is identical (same TF-IDF numbers); only the
storage format of Q and X changes:

    sparse . sparse  : Q in CSR,  X in CSR      -> inverted-index algorithm
    sparse . dense   : X in CSR,  Q dense       -> walk the non-zeros of each document
    dense  . dense   : both dense               -> plain triple loop

Each kernel is written by hand in Numba (spec: "implement the dot product for
each combination yourself") and checked against SciPy / NumPy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


def memory_mb(M):
    """Bytes actually stored: CSR = values + column indices + row pointers."""
    if sp.issparse(M):
        M = M.tocsr()
        return (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 2**20
    return np.asarray(M).nbytes / 2**20


def dense_equivalent_mb(shape, dtype=np.float32):
    return shape[0] * shape[1] * np.dtype(dtype).itemsize / 2**20


def build_tfidf(texts, max_features=10_000, min_df=2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, stop_words="english",
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts)          # rows are L2-normalised -> dot == cosine
    return vec, X.tocsr()


# ======================================================================
# sparse . sparse  (inverted index)
# ======================================================================

@njit(parallel=True, cache=True)
def _spsp_kernel(q_indptr, q_indices, q_data, inv_indptr, inv_docs, inv_vals, n_docs):
    nq = q_indptr.shape[0] - 1
    S = np.zeros((nq, n_docs), dtype=np.float32)
    for i in prange(nq):
        for p in range(q_indptr[i], q_indptr[i + 1]):        # each term t of query i
            t = q_indices[p]
            w = q_data[p]
            for r in range(inv_indptr[t], inv_indptr[t + 1]):  # each document containing t
                S[i, inv_docs[r]] += w * inv_vals[r]
    return S


def build_inverted_index(X_csr):
    """X^T in CSR form = for every term, the list of (document, weight) postings.
    Built once at indexing time (like a search engine), not per query."""
    return X_csr.T.tocsr()


def sparse_dot_sparse(Q_csr, inv_index, n_docs):
    return _spsp_kernel(Q_csr.indptr, Q_csr.indices, Q_csr.data,
                        inv_index.indptr, inv_index.indices, inv_index.data, n_docs)


# ======================================================================
# sparse . dense
# ======================================================================

@njit(parallel=True, cache=True)
def _spdn_kernel(x_indptr, x_indices, x_data, QdT):
    n_docs = x_indptr.shape[0] - 1
    nq = QdT.shape[1]
    ST = np.zeros((n_docs, nq), dtype=np.float32)       # docs x queries: contiguous writes
    for j in prange(n_docs):
        for p in range(x_indptr[j], x_indptr[j + 1]):    # non-zeros of document j
            t = x_indices[p]
            v = x_data[p]
            for i in range(nq):                         # QdT[t, :] is contiguous
                ST[j, i] += v * QdT[t, i]
    return ST


def sparse_dot_dense(X_csr, Q_dense):
    QdT = np.ascontiguousarray(Q_dense.T)
    return _spdn_kernel(X_csr.indptr, X_csr.indices, X_csr.data, QdT).T


# ======================================================================
# dense . dense
# ======================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _dndn_kernel(Q, X):
    nq, d = Q.shape
    n = X.shape[0]
    S = np.empty((nq, n), dtype=np.float32)
    for j in prange(n):
        for i in range(nq):
            s = np.float32(0.0)
            for t in range(d):
                s += Q[i, t] * X[j, t]
            S[i, j] = s
    return S


def dense_dot_dense(Q_dense, X_dense):
    return _dndn_kernel(Q_dense, X_dense)


# ======================================================================
# Library references (for correctness + "how good is our kernel?")
# ======================================================================

def ref_sparse_sparse(Q_csr, X_csr):
    return (Q_csr @ X_csr.T).toarray()          # SciPy SpGEMM, densified like ours


def ref_sparse_dense(X_csr, Q_dense):
    return (X_csr @ Q_dense.T).T                 # SciPy CSR x dense


def ref_dense_dense(Q_dense, X_dense):
    return Q_dense @ X_dense.T                   # BLAS sgemm


def random_csr(n_rows, n_cols, density, seed=0, dtype=np.float32):
    """Random sparse matrix with unit-length rows (for the density sweep)."""
    M = sp.random(n_rows, n_cols, density=density, format="csr", dtype=dtype,
                  random_state=np.random.default_rng(seed))
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).astype(dtype) @ M
