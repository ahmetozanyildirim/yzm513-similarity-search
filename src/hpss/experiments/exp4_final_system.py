"""Stage 4 (Phase 4) - final system: candidate count, stage profile, thread-pool contention,
correctness (+ float32 precision), scaling curve (+ p95 latency), all-versions comparison, the week-5
scale walls at 100M documents, the five questions, and the live demo (report section 6)."""

import json
import math
import os
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import bench, config, data
from .. import plotting as P
from ..kernels import search_numba_fused
from ..search import SearchEngine

STAGE = "4_final_system"
K = config.TOP_K


def load():
    X, Q = data.load_embeddings()
    ref = np.load(config.baseline_topk_path())
    return X, Q, ref


def chosen_reducer():
    return bench.require_stage("2b_dim_reduction")["chosen"]["method"]


def candidate_sweep(X, Q, ref, k_dims=128, candidates=(10, 20, 50, 100, 200)):
    """How many reduced-space candidates must be re-ranked to match the baseline exactly?"""
    rows = []
    for c in candidates:
        eng = SearchEngine(n_components=k_dims, n_candidates=c, reducer=chosen_reducer(), topk="numba").fit(X)
        r = bench.benchmark(eng.search, Q, K)
        chk = bench.compare_topk(*r["result"], ref["idx_20000"], ref["sc_20000"])
        rows.append({"n_candidates": c, "median_ms": r["median"] * 1e3, "all_ok": chk["all_ok"],
                     "violations": chk["violations"], "max_abs_diff": chk["max_abs_score_diff"],
                     "recall@10": bench.recall_at_k(r["result"][0], ref["idx_20000"])})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "candidates", STAGE)
    ok = df[df.all_ok]
    n_cand = int(ok.n_candidates.min()) if len(ok) else int(max(candidates))
    bench.update_summary(STAGE, n_candidates=n_cand)
    return df, n_cand


def configurations(n_cand):
    red = chosen_reducer()
    return {
        "exact 768-d + argpartition": dict(),
        "exact 768-d + parallel top-k": dict(topk="numba"),
        f"SVD-128 + re-rank({n_cand}) + argpartition": dict(n_components=128, n_candidates=n_cand, reducer=red),
        f"SVD-128 + re-rank({n_cand}) + parallel top-k": dict(n_components=128, n_candidates=n_cand, reducer=red,
                                                            topk="numba"),
    }


def stage_profile(X, Q, ref, configs, runs=20):
    """Time every stage of search(); total and stages come from the same runs."""
    rows = []
    for name, kw in configs.items():
        eng = SearchEngine(**kw).fit(X)
        for _ in range(3):
            out = eng.search(Q, K)
        parts = []
        for _ in range(runs):
            tm = {}
            t0 = time.perf_counter()
            out = eng.search(Q, K, timings=tm)
            tm["total"] = time.perf_counter() - t0
            parts.append(tm)
        med = pd.DataFrame(parts).median()
        rows.append({"config": name, "total_ms": med.pop("total") * 1e3,
                     "correct": bench.compare_topk(*out, ref["idx_20000"], ref["sc_20000"])["all_ok"],
                     **{f"{k}_ms": v * 1e3 for k, v in med.items()}})
    df = pd.DataFrame(rows).fillna(0.0)
    bench.save_csv(df, "stage_profile", STAGE)
    return df


def plot_stage_profile(stages):
    cols = [c for c in ["normalise_ms", "normalise_project_ms", "matmul_ms", "topk_ms", "topk_candidates_ms", "rerank_ms"]
            if c in stages and stages[c].sum() > 0]
    labels = {"normalise_ms": "normalise", "normalise_project_ms": "normalise + project", "matmul_ms": "matmul (BLAS)",
              "topk_ms": "top-k selection", "topk_candidates_ms": "top-k of candidates", "rerank_ms": "768-d re-rank"}
    seg = {"normalise_ms": "dask", "normalise_project_ms": "dask", "matmul_ms": "numpy", "topk_ms": "python",
           "topk_candidates_ms": "python", "rerank_ms": "final"}
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    left = np.zeros(len(stages))
    for col in cols:
        vals = stages[col].to_numpy()
        ax.barh(stages.config, vals, left=left, color=P.color(seg[col]), height=0.6, edgecolor="white",
                linewidth=2, label=labels[col])
        left += vals
    for i, tot in enumerate(stages.total_ms):
        ax.annotate(f"{tot:.2f} ms", xy=(left[i], i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9, color=P.INK_2)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("200 queries x 20,000 documents: stage times (ms, median)")
    a = stages.iloc[0]
    ax.set_title(f"Single-threaded top-k is {100 * a.topk_ms / a.total_ms:.0f}% of exact search - parallelising it "
                 f"shrinks the serial part", fontsize=10.5)
    ax.legend(ncol=3, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 1.1))
    ax.set_xlim(0, left.max() * 1.2)
    P.savefig(fig, "fig6a_final_stage_profile", STAGE)
    return fig


CHILD = r"""
import json
import numpy as np, pandas as pd
from hpss import data, bench
from hpss.search import SearchEngine
X, Q = data.load_embeddings()
configs = json.loads(%r)
rows = []
for name, kw in configs.items():
    e = SearchEngine(**kw).fit(X)
    for _ in range(3):
        e.search(Q, 10)
    parts = []
    for _ in range(20):
        tm = {}
        e.search(Q, 10, timings=tm)
        parts.append(tm)
    tot = bench.benchmark(e.search, Q, 10, warmup=0, runs=10)["median"]
    rows.append({"config": name, "total_ms": tot * 1e3, "matmul_ms": float(pd.DataFrame(parts).median()["matmul"]) * 1e3})
print("JSON" + json.dumps(rows))
"""


def thread_contention(configs):
    """Same four configurations in two fresh interpreters: idle OpenMP threads spinning (OpenMP default)
    vs sleeping (OMP_WAIT_POLICY=PASSIVE, which `import hpss` sets)."""
    rows = []
    for spinning in (True, False):
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_WAIT_POLICY", "KMP_BLOCKTIME")}
        if spinning:
            env["HPSS_NO_THREAD_FIX"] = "1"
        out = subprocess.run([sys.executable, "-c", CHILD % json.dumps(configs)], env=env, capture_output=True,
                             text=True, check=True).stdout
        for r in json.loads(out.split("JSON", 1)[1]):
            r["omp_idle_threads"] = "spinning (default)" if spinning else "sleeping (PASSIVE)"
            rows.append(r)
    df = pd.DataFrame(rows)
    bench.save_csv(df, "thread_contention", STAGE)
    return df


def choose_final(stages, configs):
    """The spec's pipeline includes dimensionality reduction: fastest correct configuration with SVD."""
    dr = stages[stages.correct & stages.config.str.startswith("SVD")].sort_values("total_ms")
    name = dr.iloc[0].config if len(dr) else "exact 768-d + parallel top-k"
    kw = configs[name]
    bench.update_summary(STAGE, final_name=name, final_config=kw, stage_profile=stages.to_dict("records"))
    return name, kw


def correctness(X, Q, ref, final_kw):
    versions = {
        "exact_blas": lambda Xs: SearchEngine().fit(Xs).search(Q, K),
        "exact_ptopk": lambda Xs: SearchEngine(topk="numba").fit(Xs).search(Q, K),
        "numba_fused": lambda Xs: search_numba_fused(np.ascontiguousarray(data.normalize_rows(Q), dtype=np.float32),
                                                     np.ascontiguousarray(data.normalize_rows(Xs), dtype=np.float32), K),
        "final": lambda Xs: SearchEngine(**final_kw).fit(Xs).search(Q, K),
    }
    sizes = [n for n in config.CORPUS_SIZES if f"idx_{n}" in ref.files]
    rows = []
    for n in sizes:
        for name, run in versions.items():
            idx, sc = run(X[:n])
            rows.append({"n_docs": n, "version": name, **bench.compare_topk(idx, sc, ref[f"idx_{n}"], ref[f"sc_{n}"])})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "correctness", STAGE)
    bench.update_summary(STAGE, correctness_all_ok=bool(df.all_ok.all()))
    return df


def synthetic(X, n, seed=0):
    """Timing-only corpus: copies of the real embeddings plus small noise."""
    rng = np.random.default_rng(seed)
    base = np.tile(X, (int(np.ceil(n / len(X))), 1))[:n]
    return (base + 0.02 * rng.standard_normal(base.shape).astype(np.float32)).astype(np.float32)


def scaling(X, Q, final_kw):
    real = [1_000, 2_000] if config.QUICK else [1_000, 2_000, 5_000, 10_000, 20_000]
    syn = [] if config.QUICK else [50_000, 100_000, 200_000]
    Qn32 = np.ascontiguousarray(data.normalize_rows(Q), dtype=np.float32)
    rows = []
    for n in real + syn:
        Xs = X[:n] if n <= len(X) else synthetic(X, n)
        kind = "real" if n <= len(X) else "synthetic"
        engines = {"exact_blas": SearchEngine().fit(Xs), "exact_ptopk": SearchEngine(topk="numba").fit(Xs),
                   "final": SearchEngine(**final_kw).fit(Xs)}
        for name, eng in engines.items():
            rb = bench.benchmark(eng.search, Q, K, warmup=3, runs=10)
            lat = [bench.benchmark(eng.search, Q[i:i + 1], K, warmup=1, runs=5)["median"] for i in range(0, 200, 4)]
            rows.append({"n_docs": n, "kind": kind, "version": name, "batch_s": rb["median"], "batch_std_s": rb["std"],
                         "latency_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
                         "index_mb": eng.memory_mb()})
        rn = bench.benchmark(search_numba_fused, Qn32, engines["exact_blas"].X, K, warmup=3, runs=10)
        rows.append({"n_docs": n, "kind": kind, "version": "numba_fused", "batch_s": rn["median"],
                     "batch_std_s": rn["std"], "latency_s": np.nan, "index_mb": engines["exact_blas"].memory_mb()})
        print(f"N={n:>7,} ({kind}) done")
    scal = pd.DataFrame(rows)
    b1 = bench.require_stage("1_baseline")
    naive = pd.DataFrame(b1["naive_per_pair_us_by_n"].items(), columns=["n_docs", "per_pair_us"]).astype(float)
    naive["batch_s"] = naive.per_pair_us * 1e-6 * naive.n_docs * len(Q)
    naive["version"], naive["kind"] = "naive", "real"
    scal = pd.concat([scal, naive[["n_docs", "kind", "version", "batch_s"]]], ignore_index=True)
    scal["n_docs"] = scal.n_docs.astype(int)
    bench.save_csv(scal, "scaling", STAGE)
    bench.save_csv(scal.assign(host=platform.node()), f"scaling_{platform.node()}", STAGE)   # bonus: machine comparison
    return scal


def slopes(scal):
    def slope(g):
        return float(np.polyfit(np.log(g.n_docs), np.log(g.batch_s), 1)[0]) if len(g) > 1 else np.nan
    rows = []
    for v in ["naive", "numba_fused", "exact_blas", "exact_ptopk", "final"]:
        g = scal[scal.version == v]
        largest_real = g[g.kind == "real"].n_docs.max()
        rows.append({"version": v, "slope_5K-20K": slope(g[(g.n_docs >= 5_000) & (g.n_docs <= 20_000)]),
                     "slope_20K-200K": slope(g[g.n_docs >= 20_000]),
                     "ms_per_1k_docs_at_largest_real": float(g[g.n_docs == largest_real].batch_s.iloc[0]) * 1e3
                     / (largest_real / 1000)})
    df = pd.DataFrame(rows)
    bench.save_csv(df, "slopes", STAGE)
    return df


NAMES = {"naive": "Naive Python (baseline)", "numba_fused": "Numba fused (prange)",
         "exact_blas": "Exact 768-d (BLAS + argpartition)", "exact_ptopk": "Exact 768-d (BLAS + parallel top-k)"}
KEYS = {"naive": "python", "numba_fused": "numba_parallel", "exact_blas": "numpy", "exact_ptopk": "numba", "final": "final"}


def plot_scaling(scal, final_name):
    names = {**NAMES, "final": f"Final: {final_name}"}
    n_l3 = 96 * 2**20 / (768 * 4)                    # N at which the 768-d float32 corpus fills the 96 MB L3
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for v in ["naive", "exact_blas", "exact_ptopk", "numba_fused", "final"]:   # order passes the CVD check
        g = scal[scal.version == v].sort_values("n_docs")
        gr, gs = g[g.kind == "real"], g[g.kind == "synthetic"]
        P.line(ax, gr.n_docs, gr.batch_s * 1e3, KEYS[v], names[v])
        if len(gs):
            gg = pd.concat([gr.tail(1), gs])
            ax.plot(gg.n_docs, gg.batch_s * 1e3, color=P.color(KEYS[v]), linestyle=":", marker=P.marker(KEYS[v]),
                    markerfacecolor="white", linewidth=1.5)
    ax.axvline(n_l3, color=P.MUTED, linewidth=0.8)
    ax.annotate("768-d corpus\n> 96 MB L3", xy=(n_l3, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -40),
                textcoords="offset points", fontsize=8.5, color=P.INK_2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("corpus size (documents, log) - dotted: synthetic"); ax.set_ylabel("time for 200 queries (ms, log)")
    f = scal[(scal.version == "final") & (scal.n_docs == 20_000)]
    if len(f):
        ax.set_title(f"(a) Final system: {f.batch_s.iloc[0] * 1e3 / 20:.2f} ms per 1K documents at 20K", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    for v in ["exact_blas", "exact_ptopk", "final"]:
        g = scal[scal.version == v].sort_values("n_docs")
        P.line(ax, g.n_docs, g.latency_s * 1e3, KEYS[v], names[v])
    ax.axvline(n_l3, color=P.MUTED, linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("corpus size (documents, log)"); ax.set_ylabel("single-query latency (ms, log)")
    lat = scal[scal.version == "final"].set_index("n_docs").latency_s
    ax.set_title(f"(b) Single query, final system: {lat.iloc[-1] * 1e3:.2f} ms at {lat.index[-1]:,} docs", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    P.savefig(fig, "fig6_scaling", STAGE)
    return fig


def all_versions(scal, final_name):
    """Same workload (200 queries x 20K) for every version built in the project."""
    b1 = bench.require_stage("1_baseline")
    s3 = bench.require_stage("3_parallelism")
    fixes = {r["version"]: r["median_s"] for r in b1["python_fixes"]}
    scale = b1["baseline"]["full_workload_s_estimate"] / fixes["naive (baseline)"]   # ladder ratio -> 200 x 20K
    n_big = scal[scal.kind == "real"].n_docs.max()
    s = scal[scal.n_docs == n_big].set_index("version").batch_s
    rows = [("Naive Python (baseline)", "python", b1["baseline"]["full_workload_s_estimate"]),
            ("Python: norms once", "python", fixes["fix 2: norms once"] * scale),
            ("Python: sum(map) + heapq", "python", fixes["fix 1+3: sum(map) + heapq"] * scale)]
    skip = ("asyncio", "fail", "ray.wait", "memmap")         # not separate versions of the search (variants / demos)
    for r in s3["summary"]:
        if not any(w in r["strategy"] for w in skip):
            rows.append((r["strategy"].replace(" (1 BLAS thread/worker)", "").replace("Phase-2 best: ", ""),
                         r["key"], r["median_s"]))
    rows += [("Exact 768-d + parallel top-k", "numba", s["exact_ptopk"]), (f"Final: {final_name}", "final", s["final"])]
    df = pd.DataFrame(rows, columns=["version", "key", "seconds"]).sort_values("seconds", ascending=False)
    df["speedup_vs_naive"] = df.seconds.iloc[0] / df.seconds
    bench.save_csv(df, "all_versions", STAGE)
    return df


def plot_all_versions(df):
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(df) + 1.2))
    labels = [bench.fmt_time(s) + (f"   ({sp:,.0f}x)" if i else "") for i, (s, sp) in enumerate(zip(df.seconds, df.speedup_vs_naive))]
    P.bars(ax, df.version.tolist(), (df.seconds * 1e3).tolist(), df.key.tolist(), log=True, horizontal=True, labels=labels)
    ax.set_xlabel("time for 200 queries x 20,000 documents (ms, log) - in brackets: speed-up over naive")
    best = df.iloc[-1]
    ax.set_title(f"Naive {df.seconds.iloc[0]:.0f} s -> {best.version}: {bench.fmt_time(best.seconds)} "
                 f"({best.speedup_vs_naive:,.0f}x)", fontsize=11.5)
    ax.set_xlim(right=df.seconds.max() * 1e3 * 60)
    P.savefig(fig, "fig7_all_versions", STAGE)
    return fig


# ---------------------------------------------------------------- float32 precision
def float32_precision(corr):
    """Why 1e-5 is a fair tolerance for float32: rounding error of a 768-term dot product vs what we measured."""
    eps = float(np.finfo(np.float32).eps)
    d = config.EMBED_DIM
    worst = corr.groupby("version").max_abs_score_diff.max()
    rows = [{"quantity": "float32 machine epsilon", "value": eps},
            {"quantity": f"typical error of a {d}-term dot product (sqrt(d) x eps)", "value": math.sqrt(d) * eps},
            {"quantity": f"worst-case bound (d x eps)", "value": d * eps},
            {"quantity": "tolerance used (spec)", "value": config.TOL}]
    rows += [{"quantity": f"measured max |score - baseline|: {v}", "value": float(x)} for v, x in worst.items()]
    df = pd.DataFrame(rows)
    bench.save_csv(df, "float32_precision", STAGE)
    return df


# ---------------------------------------------------------------- week 5: the scale walls at 100M documents
def scan_bandwidth_gbps(rows=200_000, dim=config.EMBED_DIM, repeats=5):
    """Effective GB/s of a float32 matrix @ vector (week 5, 19_scale_estimate.py) - 200K rows = 614 MB, so
    it streams from DRAM, not from the cache."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((rows, dim), dtype=np.float32)
    q = rng.standard_normal(dim, dtype=np.float32)
    X @ q
    best = min(bench.benchmark(lambda: X @ q, warmup=0, runs=1)["median"] for _ in range(repeats))
    return X.nbytes / best / 1e9


def scale_walls(k_dims=128, workers_list=(1, 8, 32, 128), mtbf_days=1000, f_ser=0.02, docs_per_s_assumed=1000):
    """The four tables of the week-5 back-of-envelope (19_scale_estimate.py). MEASURED here: scan bandwidth,
    RAM, our encoder's docs/s (stage 0, CPU). ASSUMED as on the slides: serial fraction f of the encoding job,
    1000 docs/s for a GPU worker, machine MTBF."""
    import psutil
    gbps = scan_bandwidth_gbps()
    ram = psutil.virtual_memory().total
    docs_per_s = config.N_CORPUS / config.EMBED_SECONDS
    out = {}

    rows = []                                                         # 1) memory wall
    for n in (20_000, 1_000_000, 10_000_000, 100_000_000):
        for d in (config.EMBED_DIM, k_dims):
            size = n * d * 4
            rows.append({"docs": n, "dims": d, "matrix_gb": size / 1e9, "x_this_ram": size / ram,
                         "fits_in_ram": size <= ram, "one_scan_s": size / (gbps * 1e9)})
    out["memory"] = pd.DataFrame(rows)

    n, size = 100_000_000, 100_000_000 * config.EMBED_DIM * 4         # 2) move the query, not the data
    rows = []
    for w in (32, 128):
        q_out, r_in = w * config.EMBED_DIM * 4, w * K * (8 + 4)
        rows.append({"shards": w, "rows_per_shard": n // w, "shard_gb": size / w / 1e9,
                     "scan_ms": size / w / (gbps * 1e9) * 1e3, "query_out_kb": q_out / 1e3,
                     "results_in_kb": r_in / 1e3, "moved_over_scanned": (q_out + r_in) / size})
    out["move"] = pd.DataFrame(rows)

    rows = []                                                         # 3) time wall: encoding 100M docs
    for w in workers_list:
        s = 1 / (f_ser + (1 - f_ser) / w)
        rows.append({"workers": w, "amdahl_speedup": s,
                     "ours_cpu_ideal_days": n / (docs_per_s * w) / 86400, "ours_cpu_with_f_days": n / docs_per_s / s / 86400,
                     "assumed_1000dps_with_f_hours": n / docs_per_s_assumed / s / 3600})
    out["encode"] = pd.DataFrame(rows)

    rows = []                                                         # 4) reliability wall
    enc32 = out["encode"].set_index("workers").ours_cpu_with_f_days.get(32, np.nan) * 24
    for w, h in ((1, 1), (32, 1), (128, 8), (1000, 24), (32, round(float(enc32))) if enc32 == enc32 else (32, 24)):
        rows.append({"workers": w, "job_hours": h, "p_failure": 1 - math.exp(-w * h / (mtbf_days * 24))})
    out["failure"] = pd.DataFrame(rows)

    for name, df in out.items():
        bench.save_csv(df, f"scale_{name}", STAGE)
    bench.update_summary(STAGE, scale_inputs={"scan_gbps": gbps, "ram_gb": ram / 1e9, "docs_per_s_measured": docs_per_s,
                                              "assumed_serial_fraction_f": f_ser, "assumed_docs_per_s": docs_per_s_assumed,
                                              "assumed_mtbf_days": mtbf_days})
    return out


def five_questions(final_name):
    """Week 5: 'the systems differ, the questions do not' - split / assign / move / wait / fail, for the
    engine we built (one machine) and for the same search at 100M documents (a cluster)."""
    s3 = bench.load_summary("3_parallelism")
    flaky = next((r for r in s3.get("summary", []) if "fail" in r["strategy"]), None)
    scal = pd.read_csv(config.stage_dir(STAGE) / "scaling.csv")
    fin = scal[(scal.version == "final") & (scal.kind == "real")]
    final_ms = float(fin.loc[fin.n_docs.idxmax(), "batch_s"] * 1e3) if len(fin) else float("nan")
    rows = [
        ("Split", "matmul: BLAS splits the GEMM into cache-sized blocks; top-k and re-rank: the 200 query rows over the "
                  "Numba prange threads; the corpus stays whole (61 MB, fits L3)",
         "by document: 32-128 shards of the corpus, one per machine"),
        ("Assign", "static: BLAS and OpenMP give each thread an equal block (every query costs the same)",
         "one task per shard (Dask / Ray scheduler); stateful actors keep their shard loaded"),
        ("Move", "nothing: shared memory; the corpus is read in place (no pickling)",
         f"{config.EMBED_DIM * 4 / 1e3:.1f} KB query out, {K * 12} B of (id, score) back per shard"),
        ("Wait", f"one barrier at the end of prange; the whole batch of 200 queries takes {final_ms:.2f} ms",
         "the slowest shard, then a merge of shards x k results (tree merge when the fan-in is large)"),
        ("Fail", "a crash restarts the query (milliseconds, nothing to save)",
         "retry the shard task: read-only and pure, so running it twice is safe"
         + (f" (measured: 30% failed attempts, still exact, {flaky['median_s'] * 1e3:.0f} ms)" if flaky else "")),
    ]
    df = pd.DataFrame(rows, columns=["question", f"our engine ({final_name})", "same search at 100M docs"])
    bench.save_csv(df, "five_questions", STAGE)
    return df


def demo(engine, corpus, model, text, k=5):
    t0 = time.perf_counter()
    q = data.embed_texts([text], model, show_progress=False)
    t1 = time.perf_counter()
    idx, sc = engine.search(q, k)
    t2 = time.perf_counter()
    hits = [(float(s), corpus.iloc[i].title, corpus.iloc[i].primary_category, str(corpus.iloc[i].published)[:10])
            for i, s in zip(idx[0], sc[0])]
    return {"query": text, "embed_ms": (t1 - t0) * 1e3, "search_ms": (t2 - t1) * 1e3, "hits": hits}


def finalize(stages, cont, scal, slope_df, allv, final_name):
    sl = slope_df.set_index("version")
    f = scal[scal.version == "final"].set_index("n_docs")
    s_ = scal.set_index(["version", "n_docs"]).batch_s
    a = stages.iloc[0]
    st = stages.set_index("config").total_ms
    cs = cont.set_index(["config", "omp_idle_threads"]).matmul_ms
    n20 = 20_000 if 20_000 in f.index else f.index.max()
    bench.update_summary(STAGE, slopes=slope_df.to_dict("records"), all_versions=allv.to_dict("records"),
                         final_batch_ms=float(f.loc[n20, "batch_s"] * 1e3), final_latency_ms=float(f.loc[n20, "latency_s"] * 1e3),
                         thread_contention=cont.to_dict("records"))
    caps = {
        "fig6a_final_stage_profile": (
            f"Figure 6a: Profiling our own engine shows that single-threaded argpartition takes {100 * a.topk_ms / a.total_ms:.0f}% "
            f"of the exact batched search ({a.topk_ms:.2f} of {a.total_ms:.2f} ms) - the serial fraction of Amdahl's law; a "
            f"parallel Numba top-k cuts it to {st['exact 768-d + parallel top-k']:.2f} ms and the final {final_name} engine "
            f"needs {st[final_name]:.2f} ms. Without OMP_WAIT_POLICY=PASSIVE, Numba's spinning OpenMP threads slow the "
            f"following BLAS matmul from {cs[('exact 768-d + parallel top-k', 'sleeping (PASSIVE)')]:.1f} to "
            f"{cs[('exact 768-d + parallel top-k', 'spinning (default)')]:.1f} ms (two thread pools competing for the same cores)."),
        "fig6_scaling": (
            f"Figure 6: Search time grows at most linearly with corpus size (log-log slope {sl.loc['final', 'slope_5K-20K']:.2f} "
            f"for the final engine between 5K and 20K and {sl.loc['final', 'slope_20K-200K']:.2f} up to 200K synthetic documents - "
            f"below 1 because fixed per-call costs still matter), while the fused Numba kernel, which re-reads the corpus for "
            f"every query, slows down x{s_[('numba_fused', s_.loc['numba_fused'].index.max())] / s_[('numba_fused', n20)]:.0f} "
            f"for 10x data once the corpus leaves the 96 MB L3 cache; the final engine answers 200 queries over 20K documents "
            f"in {f.loc[n20, 'batch_s'] * 1e3:.2f} ms and a single query in {f.loc[n20, 'latency_s'] * 1e3:.2f} ms."),
        "fig7_all_versions": (
            f"Figure 7: From {allv.seconds.iloc[0]:.0f} s (naive Python) to {bench.fmt_time(allv.seconds.iloc[-1])} "
            f"({allv.version.iloc[-1]}) on the same 200x20K workload, a {allv.speedup_vs_naive.iloc[-1]:,.0f}x speed-up; "
            f"Python-level fixes alone reach at most {allv[allv.key == 'python'].speedup_vs_naive.max():.0f}x - the big jump "
            f"comes from leaving the interpreter (BLAS / Numba)."),
    }
    for k, v in caps.items():
        bench.save_caption(STAGE, k, v)
    return caps


def main():
    P.setup_style()
    X, Q, ref = load()
    cands, n_cand = candidate_sweep(X, Q, ref)
    print(cands)
    configs = configurations(n_cand)
    stages = stage_profile(X, Q, ref, configs)
    plot_stage_profile(stages)
    cont = thread_contention(configs)
    final_name, final_kw = choose_final(stages, configs)
    print("final:", final_name, final_kw)
    corr = correctness(X, Q, ref, final_kw)
    print(corr)
    print(float32_precision(corr))
    scal = scaling(X, Q, final_kw)
    slope_df = slopes(scal)
    plot_scaling(scal, final_name)
    allv = all_versions(scal, final_name)
    plot_all_versions(allv)
    for name, df in scale_walls().items():
        print(name, df, sep="\n")
    print(five_questions(final_name))
    for v in finalize(stages, cont, scal, slope_df, allv, final_name).values():
        print(v, "\n")


if __name__ == "__main__":
    main()
