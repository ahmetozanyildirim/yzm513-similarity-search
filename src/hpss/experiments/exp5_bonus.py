"""Stage 5 (bonus) - roofline model, approximate nearest neighbours (own LSH + FAISS HNSW) and the
laptop-vs-Colab comparison (report appendix)."""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange

from .. import bench, config, data
from .. import plotting as P
from ..kernels import search_numba_fused
from ..search import SearchEngine, search_numpy, search_numpy_per_query, topk_from_scores

STAGE = "5_bonus"
K = config.TOP_K


def load():
    X, Q = data.load_embeddings()
    Xn = np.ascontiguousarray(data.normalize_rows(X), dtype=np.float32)
    Qn = np.ascontiguousarray(data.normalize_rows(Q), dtype=np.float32)
    return X, Q, Xn, Qn


# ---------------------------------------------------------------- roofline
@njit(parallel=True, fastmath=True, cache=True)
def triad(a, b, c, s):
    """STREAM triad a = b + s*c: 2 reads + 1 write per element."""
    for i in prange(a.shape[0]):
        a[i] = b[i] + s * c[i]


def bandwidth(total_mb, reps=5):
    n = int(total_mb * 2**20 / 4 / 3)
    a, b, c = (np.ones(n, np.float32) for _ in range(3))
    res = bench.benchmark(lambda: [triad(a, b, c, 1.5) for _ in range(reps)], warmup=2, runs=5)
    return 3 * 4 * n * reps / res["median"] / 1e9


def machine_roofs(n=4096):
    A = np.random.default_rng(0).standard_normal((n, n), dtype=np.float32)
    peak = 2 * n**3 / bench.benchmark(np.matmul, A, A, warmup=2, runs=5)["median"] / 1e9
    with bench.blas_threads(1):
        peak1 = 2 * n**3 / bench.benchmark(np.matmul, A, A, warmup=1, runs=5)["median"] / 1e9
    roofs = {"peak_gflops": peak, "peak_gflops_1thread": peak1,
             "bw_dram_gbs": bandwidth(512 if config.QUICK else 3 * 1024),     # does not fit in the 96 MB L3
             "bw_l3_gbs": bandwidth(48)}                                       # fits in L3
    bench.update_summary(STAGE, roofline=roofs)
    return roofs


def _bytes_gemm(nq, N, d):
    return 4 * (N * d + nq * d + nq * N)            # compulsory traffic: corpus + queries + score matrix


def roofline_points(X, Q, Xn, Qn):
    nq, N, d = len(Qn), len(Xn), Xn.shape[1]
    flops = 2 * nq * N * d
    final_kw = bench.require_stage("4_final_system")["final_config"]
    pts = {}
    t = bench.benchmark(lambda: Qn @ Xn.T)["median"]
    pts["GEMM batch (BLAS, all threads)"] = ("numpy", flops / t / 1e9, flops / _bytes_gemm(nq, N, d))
    t = bench.benchmark(lambda: [Xn @ q for q in Qn])["median"]
    pts["GEMV per query (BLAS)"] = ("python", flops / t / 1e9, flops / (4 * nq * N * d))
    eng = SearchEngine(**final_kw).fit(X)
    t = bench.benchmark(eng.search, Q, K)["median"]
    kd, c = final_kw.get("n_components") or d, final_kw.get("n_candidates", 0)
    fl = 2 * nq * N * kd + 2 * nq * c * d
    pts["Final system (whole search)"] = ("final", fl / t / 1e9, fl / (_bytes_gemm(nq, N, kd) + 4 * nq * c * d))
    with bench.blas_threads(1):
        t = bench.benchmark(lambda: Qn @ Xn.T)["median"]
    pts["GEMM batch (BLAS, 1 thread)"] = ("", flops / t / 1e9, flops / _bytes_gemm(nq, N, d))
    t = bench.benchmark(search_numba_fused, Qn, Xn, K)["median"]
    pts["Numba fused (prange)"] = ("", flops / t / 1e9, flops / (4 * nq * N * d))
    df = pd.DataFrame([(k, *v) for k, v in pts.items()], columns=["kernel", "key", "gflops", "ai"])
    bench.save_csv(df, "roofline_points", STAGE)
    return df


def plot_roofline(roofs, pts):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ai = np.logspace(-1.5, 3, 200)
    ax.plot(ai, np.minimum(roofs["peak_gflops"], roofs["bw_dram_gbs"] * ai), color=P.INK_2, linewidth=1.6,
            label=f"DRAM roof ({roofs['bw_dram_gbs']:.0f} GB/s)")
    ax.plot(ai, np.minimum(roofs["peak_gflops"], roofs["bw_l3_gbs"] * ai), color=P.MUTED, linewidth=1.2,
            label=f"L3 roof ({roofs['bw_l3_gbs']:.0f} GB/s)")
    ax.axhline(roofs["peak_gflops_1thread"], color=P.AXIS, linewidth=1)
    ax.annotate(f"1-thread peak ~ {roofs['peak_gflops_1thread']:.0f} GFLOP/s", xy=(ai[0], roofs["peak_gflops_1thread"]),
                xytext=(2, 3), textcoords="offset points", fontsize=8.5, color=P.INK_2)
    ax.annotate(f"all-core peak ~ {roofs['peak_gflops']:.0f} GFLOP/s", xy=(ai[-1], roofs["peak_gflops"]), xytext=(-4, 4),
                textcoords="offset points", fontsize=8.5, color=P.INK_2, ha="right")
    for _, r in pts[pts.key != ""].iterrows():          # 3 points so every colour pair stays distinguishable
        ax.plot(r.ai, r.gflops, marker=P.marker(r.key), color=P.color(r.key), markersize=9, markeredgecolor="white",
                linestyle="none", label=r.kernel)
        ax.annotate(f"{r.gflops:.0f} GFLOP/s", xy=(r.ai, r.gflops), xytext=(7, -3), textcoords="offset points",
                    fontsize=8.5, color=P.INK_2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte, log)"); ax.set_ylabel("achieved GFLOP/s (log)")
    ax.set_title("Batched GEMM is compute-bound; per-query GEMV is bandwidth-bound", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    P.savefig(fig, "fig8_roofline", STAGE)
    return fig


# ---------------------------------------------------------------- ANN
class LSHIndex:
    """Random-hyperplane LSH (SimHash): n_tables tables of n_bits-bit codes; candidates are re-ranked exactly."""

    def __init__(self, n_bits=12, n_tables=8, dim=768, seed=0):
        self.b, self.L = n_bits, n_tables
        self.R = np.random.default_rng(seed).standard_normal((n_tables, dim, n_bits)).astype(np.float32)
        self.pow2 = (1 << np.arange(n_bits)).astype(np.int64)

    def _codes(self, A, t):
        return ((A @ self.R[t]) > 0).astype(np.int64) @ self.pow2           # one b-bit integer per row

    def fit(self, Xa):
        self.X = Xa
        self.tables = []
        for t in range(self.L):
            codes = self._codes(Xa, t)
            order = np.argsort(codes, kind="stable")                         # equal codes become neighbours
            self.tables.append((codes[order], order))
        return self

    def search(self, Qa, k):
        idx = np.empty((len(Qa), k), np.int64)
        val = np.empty((len(Qa), k), np.float32)
        qcodes = [self._codes(Qa, t) for t in range(self.L)]
        n_cand = 0
        for i, q in enumerate(Qa):
            parts = []
            for t, (sorted_codes, order) in enumerate(self.tables):
                lo, hi = np.searchsorted(sorted_codes, [qcodes[t][i], qcodes[t][i] + 1])
                parts.append(order[lo:hi])
            cand = np.unique(np.concatenate(parts))
            n_cand += len(cand)
            if len(cand) < k:                                               # empty buckets: brute force
                cand = np.arange(len(self.X))
            ci, cv = topk_from_scores(self.X[cand] @ q, k)
            idx[i], val[i] = cand[ci[0]], cv[0]
        self.mean_candidates = n_cand / len(Qa)
        return idx, val


def ann_experiment(Xn, Qn):
    exact_idx, _ = search_numpy(Qn, Xn, K)
    rows = []
    for b, L in [(12, 8)] if config.QUICK else [(16, 4), (14, 8), (12, 8), (12, 16), (10, 16), (8, 16)]:
        lsh = LSHIndex(b, L, Xn.shape[1]).fit(Xn)
        r = bench.benchmark(lsh.search, Qn, K, warmup=1, runs=5)
        rows.append({"method": "LSH (ours)", "param": f"b={b}, L={L}", "ms_per_query": r["median"] / len(Qn) * 1e3,
                     "recall@10": bench.recall_at_k(r["result"][0], exact_idx),
                     "candidates_frac": lsh.mean_candidates / len(Xn)})
    try:
        import faiss
        import time
        faiss.omp_set_num_threads(1)                                        # fair: one thread per query
        t0 = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(Xn.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(Xn)
        bench.update_summary(STAGE, hnsw_build_s=time.perf_counter() - t0)
        for ef in [16, 32, 64, 128, 256]:
            hnsw.hnsw.efSearch = ef
            r = bench.benchmark(hnsw.search, Qn, K, warmup=1, runs=5)
            rows.append({"method": "HNSW (FAISS)", "param": f"efSearch={ef}", "ms_per_query": r["median"] / len(Qn) * 1e3,
                         "recall@10": bench.recall_at_k(r["result"][1], exact_idx), "candidates_frac": np.nan})
    except ImportError:
        print("faiss-cpu not installed - HNSW skipped")
    with bench.blas_threads(1):
        r = bench.benchmark(search_numpy_per_query, Qn, Xn, K)
    rows.append({"method": "Exact (BLAS, 1 thread)", "param": "", "ms_per_query": r["median"] / len(Qn) * 1e3,
                 "recall@10": 1.0, "candidates_frac": 1.0})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "ann", STAGE)
    return df


def plot_ann(ann):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for meth, key in [("LSH (ours)", "lsh"), ("HNSW (FAISS)", "dask"), ("Exact (BLAS, 1 thread)", "numpy")]:
        g = ann[ann.method == meth].sort_values("ms_per_query")
        if len(g):
            P.line(ax, g.ms_per_query, g["recall@10"], key, meth)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("time per query (ms, log, single thread)"); ax.set_ylabel("recall@10 vs exact search")
    h = ann[(ann.method == "HNSW (FAISS)") & (ann["recall@10"] >= 0.95)].sort_values("ms_per_query").head(1)
    exact = ann[ann.method.str.startswith("Exact")].ms_per_query.iloc[0]
    ax.set_title(f"HNSW at recall >= 0.95 is {exact / h.ms_per_query.iloc[0]:.0f}x faster than exact search" if len(h)
                 else "ANN: speed-accuracy trade-off", fontsize=11)
    ax.legend(loc="lower right")
    P.savefig(fig, "fig9_ann_tradeoff", STAGE)
    return fig


# ---------------------------------------------------------------- machines
def machine_comparison():
    files = sorted(glob.glob(str(config.stage_dir("4_final_system") / "scaling_*.csv")))
    hosts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(hosts) and hosts.host.nunique() > 1:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, (h, g) in enumerate(hosts[hosts.version == "final"].groupby("host")):
            g = g.sort_values("n_docs")
            ax.plot(g.n_docs, g.batch_s * 1e3, marker=P.MARKERS[i], color=P.PALETTE[i], label=h)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("corpus size"); ax.set_ylabel("time for 200 queries (ms)")
        ax.set_title("Final system on different machines", fontsize=11); ax.legend()
        P.savefig(fig, "fig10_machines", STAGE)
    return hosts


def finalize(roofs, pts, ann):
    p = pts.set_index("kernel")
    g = p.loc["GEMM batch (BLAS, all threads)"]
    v = p.loc["GEMV per query (BLAS)"]
    bench.save_caption(STAGE, "fig8_roofline",
        f"Figure 8 (bonus): Batched GEMM search runs at {g.gflops:.0f} GFLOP/s (AI~{g.ai:.0f} FLOP/B, "
        f"{100 * g.gflops / roofs['peak_gflops']:.0f}% of the measured {roofs['peak_gflops']:.0f} GFLOP/s compute roof) while "
        f"per-query GEMV reaches only {v.gflops:.0f} GFLOP/s at AI=0.5, bounded by memory bandwidth "
        f"({roofs['bw_l3_gbs']:.0f} GB/s L3, {roofs['bw_dram_gbs']:.0f} GB/s DRAM): batching queries is what moves the search "
        f"from memory-bound to compute-bound.")
    h = ann[ann.method == "HNSW (FAISS)"]
    if len(h):
        best = h.sort_values("recall@10").iloc[-1]
        exact = ann[ann.method.str.startswith("Exact")].ms_per_query.iloc[0]
        bench.save_caption(STAGE, "fig9_ann_tradeoff",
            f"Figure 9 (bonus): HNSW reaches recall@10 {best['recall@10']:.3f} at {best.ms_per_query:.3f} ms per query vs "
            f"{exact:.3f} ms for exact single-threaded search; our SimHash LSH needs to scan a large fraction of the corpus for "
            f"high recall in 768 dimensions. At 20K documents exact search is already sub-millisecond, so ANN only pays off "
            f"at much larger N.")


def main():
    P.setup_style()
    X, Q, Xn, Qn = load()
    roofs = machine_roofs()
    print(roofs)
    pts = roofline_points(X, Q, Xn, Qn)
    print(pts)
    plot_roofline(roofs, pts)
    ann = ann_experiment(Xn, Qn)
    print(ann)
    plot_ann(ann)
    machine_comparison()
    finalize(roofs, pts, ann)


if __name__ == "__main__":
    main()
