"""Phase 2B - dimensionality reduction: our own power-iteration SVD vs. library methods.

All methods return the same thing (see `reduce`):
    Z          (m x k)  coordinates of the rows in the reduced space
    components (k x n)  the basis (right singular vectors / principal axes)
    mean       (n,) or None  PCA centres the data, SVD does not
so that  X  ~=  Z @ components (+ mean)  and the reconstruction error is comparable.
"""

from __future__ import annotations

import time

import numpy as np


# ======================================================================
# Our implementation: block power iteration (a.k.a. subspace / simultaneous iteration)
# ======================================================================

def power_iteration_svd(X, k, max_iter=100, tol=1e-6, seed=0, return_history=False):
    """Top-k singular triplets of X (m x n) by block power iteration.

    Idea: repeatedly multiplying a random block V (n x k) by X^T X amplifies the
    directions with large singular values (sigma_i^2 grows fastest); QR keeps the
    k columns orthonormal so they don't all collapse onto the top vector.

        V <- qr( X^T (X V) )          # never forms X^T X explicitly

    Convergence: stop when the singular-value estimates sigma_i = ||X v_i||
    change by less than `tol` (relative) between two iterations.
    Returns U (m x k), S (k,), Vt (k x n)  [+ history of the relative change].
    """
    m, n = X.shape
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n, k)).astype(X.dtype))
    s_old = np.zeros(k)
    history = []
    for it in range(max_iter):
        XV = X @ V                                   # m x k
        s_new = np.linalg.norm(XV, axis=0)           # current singular value estimates
        change = float(np.max(np.abs(s_new - s_old) / np.maximum(s_new, 1e-30)))
        history.append(change)
        if change < tol:
            break
        s_old = s_new
        V, _ = np.linalg.qr(X.T @ XV)                # n x k, orthonormal columns
    XV = X @ V
    S = np.linalg.norm(XV, axis=0)
    U = XV / S
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], V[:, order].T
    if return_history:
        return U, S, Vt, {"iterations": it + 1, "history": history}
    return U, S, Vt


# ======================================================================
# A common interface for every method
# ======================================================================

def reduce(X, k, method="truncated_svd", seed=0, max_iter=100, tol=1e-6):
    """Fit a k-dimensional reduction of X; returns dict(Z, components, mean, info)."""
    info = {}
    if method == "power_iteration":
        U, S, Vt, h = power_iteration_svd(X, k, max_iter=max_iter, tol=tol, seed=seed,
                                          return_history=True)
        Z, comps, mean = U * S, Vt, None
        info["iterations"] = h["iterations"]
    elif method == "numpy_svd":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)       # full SVD, then truncate
        Z, comps, mean = U[:, :k] * S[:k], Vt[:k], None
    elif method == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD
        tsvd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=seed)
        Z = tsvd.fit_transform(X)
        comps, mean = tsvd.components_, None
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k, random_state=seed)          # svd_solver="auto"
        Z = pca.fit_transform(X)
        comps, mean = pca.components_, pca.mean_
        info["solver"] = getattr(pca, "_fit_svd_solver", "?")
    else:
        raise ValueError(method)
    return {"Z": Z, "components": comps, "mean": mean, "info": info}


def reconstruction_error(X, fit):
    """Frobenius norm ||X - X_hat||_F and the same relative to ||X||_F."""
    X_hat = fit["Z"] @ fit["components"]
    if fit["mean"] is not None:
        X_hat = X_hat + fit["mean"]
    err = float(np.linalg.norm(X - X_hat))
    return err, err / float(np.linalg.norm(X))


def optimal_error(S_full, k, X_norm):
    """Eckart-Young: the best possible rank-k error is sqrt(sum of the dropped sigma^2)."""
    err = float(np.sqrt(np.sum(S_full[k:] ** 2)))
    return err, err / X_norm


def time_reduce(X, k, method, **kw):
    t0 = time.perf_counter()
    fit = reduce(X, k, method, **kw)
    return time.perf_counter() - t0, fit
