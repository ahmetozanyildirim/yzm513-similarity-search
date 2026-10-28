"""Stage 2B (Phase 2B) - dimensionality reduction: our power-iteration SVD vs. library methods (section 3.2)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import bench, config, data, svd
from .. import plotting as P
from ..search import SearchEngine, search_numpy

STAGE = "2b_dim_reduction"
METHODS = ["power_iteration", "truncated_svd", "pca", "numpy_svd"]
NAMES = {"power_iteration": "our power iteration", "numpy_svd": "NumPy full SVD",
         "truncated_svd": "sklearn TruncatedSVD", "pca": "sklearn PCA"}
K_LIST = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700]
K_OURS = [10, 25, 50, 100, 200, 300, 400]          # ours is very slow for large k


def load():
    X_raw, Q_raw = data.load_embeddings()
    return data.normalize_rows(X_raw.astype(np.float64)), data.normalize_rows(Q_raw.astype(np.float64))


def full_svd(X):
    return np.linalg.svd(X, full_matrices=False)      # ground truth: U, S, Vt


def verify_power_iteration(X, S_np, Vt_np, k=50):
    """Our SVD vs NumPy: singular values, subspace overlap, reconstruction error vs Eckart-Young."""
    U, S, Vt, hist = svd.power_iteration_svd(X, k, return_history=True)
    rel = np.abs(S - S_np[:k]) / S_np[:k]
    overlap = np.linalg.svd(Vt @ Vt_np[:k].T, compute_uv=False)       # cosines of principal angles
    err = float(np.linalg.norm(X - (U * S) @ Vt))
    opt, _ = svd.optimal_error(S_np, k, float(np.linalg.norm(X)))
    out = {"k": k, "iterations": hist["iterations"], "sv_rel_err_median": float(np.median(rel)),
           "sv_rel_err_max": float(rel.max()), "subspace_overlap_min": float(overlap.min()),
           "recon_error_ours": err, "recon_error_optimal": opt, "excess_pct": 100 * (err / opt - 1)}
    bench.update_summary(STAGE, power_iteration_check=out)
    return out, hist, S


def plot_convergence(hist, S_np, S_ours):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].semilogy(np.arange(1, len(hist["history"]) + 1), hist["history"], color=P.color("power_iteration"), marker=".")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("max relative change of singular values")
    axes[0].set_title("Power-iteration convergence (k=50)", fontsize=11)
    axes[1].plot(np.arange(1, 101), S_np[:100], color=P.color("numpy"), label="NumPy (full SVD)")
    axes[1].plot(np.arange(1, len(S_ours) + 1), S_ours, linestyle="none", marker="o", markersize=4,
                 color=P.color("power_iteration"), label=f"ours (k={len(S_ours)})")
    axes[1].set_xlabel("i"); axes[1].set_ylabel("singular value"); axes[1].set_title("First 100 singular values", fontsize=11)
    axes[1].legend()
    P.savefig(fig, "fig3b_power_iteration_check", STAGE)
    return fig


def time_vs_k(X):
    k_list, k_ours = ([10, 50, 100], [10, 50]) if config.QUICK else (K_LIST, K_OURS)
    full = bench.benchmark(np.linalg.svd, X, full_matrices=False, warmup=2, runs=7)
    rows = [{"method": "numpy_svd", "k": k, "median_s": full["median"], "std_s": full["std"]} for k in k_list]
    for method in ["power_iteration", "truncated_svd", "pca"]:
        for k in (k_ours if method == "power_iteration" else k_list):
            r = bench.benchmark(svd.reduce, X, k, method, warmup=1, runs=5)
            rows.append({"method": method, "k": k, "median_s": r["median"], "std_s": r["std"],
                         **{f"info_{a}": b for a, b in r["result"]["info"].items()}})
        print(method, "done")
    t_k = pd.DataFrame(rows)
    bench.save_csv(t_k, "time_vs_k", STAGE)
    ts = t_k[t_k.method == "truncated_svd"].set_index("k").median_s
    lose = ts[ts > full["median"]]
    k_cross = int(lose.index.min()) if len(lose) else None
    bench.update_summary(STAGE, full_svd={"median_s": full["median"], "min_s": full["min"], "max_s": full["max"]},
                         k_crossover_truncated_vs_full=k_cross)
    return t_k, full, k_cross


def time_vs_m(X, k=100):
    rows = []
    for m in (config.CORPUS_SIZES[:2] if config.QUICK else config.CORPUS_SIZES):
        for method in ["power_iteration", "numpy_svd", "truncated_svd", "pca"]:
            r = bench.benchmark(svd.reduce, X[:m], k, method, warmup=1, runs=5)
            rows.append({"method": method, "m": m, "median_s": r["median"], "std_s": r["std"]})
    t_m = pd.DataFrame(rows)
    bench.save_csv(t_m, "time_vs_m", STAGE)
    return t_m


def errors_and_memory(X, S_np):
    rows = []
    for k in ([50] if config.QUICK else [50, 100, 200]):
        for method in ["power_iteration", "numpy_svd", "truncated_svd", "pca"]:
            peak, fit = bench.peak_memory_mb(svd.reduce, X, k, method)
            e, rel = svd.reconstruction_error(X, fit)
            rows.append({"k": k, "method": method, "frobenius": e, "relative": rel, "peak_mem_mb": peak})
        e, rel = svd.optimal_error(S_np, k, float(np.linalg.norm(X)))
        rows.append({"k": k, "method": "optimal (Eckart-Young)", "frobenius": e, "relative": rel, "peak_mem_mb": np.nan})
    errs = pd.DataFrame(rows)
    bench.save_csv(errs, "errors_memory", STAGE)
    return errs


def mean_direction(X, S_np, Vt_np):
    """Are the embeddings centred? Does the first singular vector point at the mean?"""
    mu = X.mean(axis=0)
    out = {"mean_norm": float(np.linalg.norm(mu)),
           "cos_v1_mean": float(abs(Vt_np[0] @ mu) / np.linalg.norm(mu)),
           "sigma1_energy": float(S_np[0] ** 2 / (S_np ** 2).sum())}
    bench.update_summary(STAGE, mean_direction=out)
    return out


def search_quality(X, Qn):
    """recall@10 of k-dim search vs exact 768-d search, with and without 100-candidate re-ranking."""
    X32, Q32 = X.astype(np.float32), Qn.astype(np.float32)
    exact_idx, _ = search_numpy(Q32, X32, config.TOP_K)
    t_exact = bench.benchmark(search_numpy, Q32, X32, config.TOP_K)["median"]
    rows = []
    for k in ([32, 128] if config.QUICK else [16, 32, 64, 96, 128, 192, 256, 384]):
        plain = SearchEngine(n_components=k, n_candidates=config.TOP_K).fit(X32)     # candidates = k -> no re-rank
        rr = SearchEngine(n_components=k, n_candidates=100).fit(X32)
        rp, rr_ = bench.benchmark(plain.search, Q32, config.TOP_K), bench.benchmark(rr.search, Q32, config.TOP_K)
        rows.append({"k": k, "recall_plain": bench.recall_at_k(rp["result"][0], exact_idx),
                     "recall_rerank100": bench.recall_at_k(rr_["result"][0], exact_idx),
                     "search_plain_ms": rp["median"] * 1e3, "search_rerank_ms": rr_["median"] * 1e3})
    rec = pd.DataFrame(rows)
    rec["exact_768_ms"] = t_exact * 1e3
    bench.save_csv(rec, "search_quality", STAGE)
    return rec, exact_idx


def reducer_for_search(X, Qn, exact_idx, k=128, n_candidates=100):
    """Same k and candidates, three reducers, checked against the naive baseline (tolerance 1e-5)."""
    ref = np.load(config.baseline_topk_path())
    X32, Q32 = X.astype(np.float32), Qn.astype(np.float32)
    rows = []
    for method in ["truncated_svd", "power_iteration", "pca"]:
        eng = SearchEngine(n_components=k, n_candidates=n_candidates, reducer=method).fit(X32)
        idx, sc = eng.search(Q32, config.TOP_K)
        chk = bench.compare_topk(idx, sc, ref["idx_20000"], ref["sc_20000"])
        rows.append({"method": method,
                     "fit_s": bench.benchmark(svd.reduce, X, k, method, warmup=1, runs=5)["median"],
                     "recall@10": bench.recall_at_k(idx, exact_idx),
                     "same_as_baseline": chk["all_ok"], "wrong_ranks": chk["violations"]})
    red = pd.DataFrame(rows)
    bench.save_csv(red, "reducer_search", STAGE)
    return red


def plot_figure3(t_k, t_m, errs, rec, k_cross):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax = axes[0, 0]
    for m in METHODS:
        g = t_k[t_k.method == m]
        P.line(ax, g.k, g.median_s, m, NAMES[m])
    ax.set_yscale("log"); ax.set_xlabel("components k"); ax.set_ylabel("time (s, log)")
    ax.set_title("(a) Time vs k (20,000 x 768)" + (f": full SVD overtakes TruncatedSVD at k~{k_cross}" if k_cross else ""),
                 fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax = axes[0, 1]
    for m in METHODS:
        g = t_m[t_m.method == m]
        P.line(ax, g.m, g.median_s, m, NAMES[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(t_m.m.unique()), [f"{v // 1000}K" for v in sorted(t_m.m.unique())])
    ax.set_xlabel("rows m"); ax.set_ylabel("time (s, log)")
    ax.set_title("(b) Time vs matrix size (k = 100)", fontsize=10.5)
    ax = axes[1, 0]
    ek = errs[~errs.method.str.startswith("optimal")]
    for m in METHODS:
        g = ek[ek.method == m]
        P.line(ax, g.k, g.relative, m, NAMES[m])
    ax.set_xlabel("components k"); ax.set_ylabel("relative Frobenius error")
    ax.set_xticks(sorted(ek.k.unique()))
    ax.set_title("(c) Reconstruction error: all methods at the optimum", fontsize=10.5)
    ax = axes[1, 1]
    P.line(ax, rec.k, rec.recall_plain, "truncated_svd", "k-dim search only")
    P.line(ax, rec.k, rec.recall_rerank100, "rerank", "k-dim + exact 768-d re-rank of 100")
    ax.set_ylim(0, 1.05); ax.set_xlabel("components k"); ax.set_ylabel("recall@10 vs exact 768-d search")
    r = rec.set_index("k").loc[128]
    ax.set_title(f"(d) k=128: recall {r.recall_rerank100:.3f} with re-rank, {r.recall_plain:.2f} without", fontsize=10.5)
    ax.legend(loc="lower right")
    P.savefig(fig, "fig3_dim_reduction", STAGE)
    return fig


def finalize(t_k, full, errs, rec, red, k_cross):
    tk = t_k.pivot(index="k", columns="method", values="median_s")
    e100 = errs[errs.k == errs.k.unique()[min(1, errs.k.nunique() - 1)]].set_index("method").relative
    r128 = rec.set_index("k").loc[128]
    chosen = {"method": "truncated_svd", "k": 128, "n_candidates": 100}
    bench.update_summary(STAGE, chosen=chosen, time_k100=tk.loc[100].to_dict() if 100 in tk.index else None,
                         rel_error=e100.to_dict(), recall_k128=r128.to_dict(), reducer_search=red.to_dict("records"),
                         pca_solver=str(t_k[t_k.method == "pca"].filter(like="info_solver").iloc[0, 0]))
    pca_recall = red.set_index("method").loc["pca", "recall@10"]
    cap = (f"Figure 3: Truncated (randomised) SVD reaches the optimal Eckart-Young error at k=100 "
           f"({e100['truncated_svd']:.3f} relative) in {tk.loc[100, 'truncated_svd']:.2f} s versus {full['median']:.2f} s for "
           f"the full SVD and {tk.loc[100, 'power_iteration']:.2f} s for our power iteration"
           + (f"; the full SVD overtakes it from k~{k_cross} on" if k_cross else "")
           + f"; at k=128, re-ranking 100 reduced-space candidates in full 768-d recovers {r128.recall_rerank100:.1%} of the "
           f"exact top-10 (only {r128.recall_plain:.0%} without re-ranking), whereas PCA - fastest and lowest-error - reaches "
           f"only {pca_recall:.1%} because centring changes the cosine geometry the search relies on.")
    bench.save_caption(STAGE, "fig3_dim_reduction", cap)
    return cap


def main():
    P.setup_style()
    X, Qn = load()
    U_np, S_np, Vt_np = full_svd(X)
    check, hist, S_ours = verify_power_iteration(X, S_np, Vt_np)
    print(check)
    plot_convergence(hist, S_np, S_ours)
    t_k, full, k_cross = time_vs_k(X)
    t_m = time_vs_m(X)
    errs = errors_and_memory(X, S_np)
    print(mean_direction(X, S_np, Vt_np))
    rec, exact_idx = search_quality(X, Qn)
    red = reducer_for_search(X, Qn, exact_idx)
    print(red)
    plot_figure3(t_k, t_m, errs, rec, k_cross)
    print(finalize(t_k, full, errs, rec, red, k_cross))


if __name__ == "__main__":
    main()
