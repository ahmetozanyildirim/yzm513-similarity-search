"""Stage 2A (Phase 2A) - sparse vs dense representations with our own dot-product kernels (section 3.1)."""

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from .. import bench, config, data
from .. import plotting as P
from .. import sparse_dense as sd

STAGE = "2a_sparse_dense"
SIZES = config.CORPUS_SIZES


def load():
    corpus, queries = data.load_corpus(), data.load_queries()
    X, Q = data.load_embeddings()
    vec, X_tfidf = sd.build_tfidf(corpus["abstract"])
    Q_tfidf = vec.transform(queries["text"]).astype(np.float32).tocsr()
    return corpus, queries, data.normalize_rows(X), data.normalize_rows(Q), vec, X_tfidf, Q_tfidf


def tfidf_stats(X_tfidf, Q_tfidf):
    N, V = X_tfidf.shape
    out = {"vocab": V, "nnz": int(X_tfidf.nnz), "density": X_tfidf.nnz / (N * V),
           "nnz_per_doc": X_tfidf.nnz / N, "nnz_per_query": Q_tfidf.nnz / Q_tfidf.shape[0]}
    bench.update_summary(STAGE, tfidf=out)
    return out


def memory_table(X_tfidf, X_emb):
    rows = [{"n_docs": n, "tfidf_csr_mb": sd.memory_mb(X_tfidf[:n]),
             "tfidf_dense_mb": sd.dense_equivalent_mb(X_tfidf[:n].shape),
             "embedding_dense_mb": sd.memory_mb(X_emb[:n])} for n in SIZES]
    mem = pd.DataFrame(rows)
    mem["dense_tfidf_vs_csr"] = mem.tfidf_dense_mb / mem.tfidf_csr_mb
    bench.save_csv(mem, "memory", STAGE)
    return mem


def check_kernels(X_tfidf, Q_tfidf, n0=2_000):
    """Our three kernels vs SciPy / NumPy (also triggers Numba compilation)."""
    Xs, Xd, Qd = X_tfidf[:n0], X_tfidf[:n0].toarray(), Q_tfidf.toarray()
    inv = sd.build_inverted_index(Xs)
    return {
        "sparse.sparse": float(np.abs(sd.sparse_dot_sparse(Q_tfidf, inv, n0) - sd.ref_sparse_sparse(Q_tfidf, Xs)).max()),
        "sparse.dense": float(np.abs(sd.sparse_dot_dense(Xs, Qd) - sd.ref_sparse_dense(Xs, Qd)).max()),
        "dense.dense": float(np.abs(sd.dense_dot_dense(Qd, Xd) - sd.ref_dense_dense(Qd, Xd)).max()),
    }


def timing(X_tfidf, Q_tfidf, X_emb, Q_emb):
    """All kernels single-threaded (the question here is representation, not parallelism)."""
    numba.set_num_threads(1)
    Qd = Q_tfidf.toarray()
    rows = []
    sizes = SIZES[:2] if config.QUICK else SIZES
    with bench.blas_threads(1):
        for n in sizes:
            Xs = X_tfidf[:n]
            Xd = Xs.toarray()
            t_inv = bench.benchmark(sd.build_inverted_index, Xs, warmup=1, runs=5)
            inv = t_inv["result"]
            cases = {
                ("sparse.sparse", "ours"): (sd.sparse_dot_sparse, (Q_tfidf, inv, n)),
                ("sparse.sparse", "scipy"): (sd.ref_sparse_sparse, (Q_tfidf, Xs)),
                ("sparse.dense", "ours"): (sd.sparse_dot_dense, (Xs, Qd)),
                ("sparse.dense", "scipy"): (sd.ref_sparse_dense, (Xs, Qd)),
                ("dense.dense", "ours"): (sd.dense_dot_dense, (Qd, Xd)),
                ("dense.dense", "numpy"): (sd.ref_dense_dense, (Qd, Xd)),
                ("embedding 768-d", "numpy"): (sd.ref_dense_dense, (Q_emb, X_emb[:n])),
            }
            for (combo, impl), (fn, args) in cases.items():
                r = bench.benchmark(fn, *args, warmup=1, runs=5)
                rows.append({"n_docs": n, "combo": combo, "impl": impl, "median_s": r["median"],
                             "std_s": r["std"], "index_build_s": t_inv["median"]})
            print(f"N={n:>6,} done")
            del Xd
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    times = pd.DataFrame(rows)
    bench.save_csv(times, "times", STAGE)
    return times


def plot_memory_time(mem, times):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    P.line(ax, mem.n_docs, mem.tfidf_dense_mb, "numpy", "TF-IDF, dense (zeros stored)")
    P.line(ax, mem.n_docs, mem.embedding_dense_mb, "embedding", "Embedding 768-d, dense")
    P.line(ax, mem.n_docs, mem.tfidf_csr_mb, "sparse", "TF-IDF, CSR (sparse)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("corpus size (documents)"); ax.set_ylabel("memory (MB, log)")
    r20 = mem.iloc[-1]
    ax.set_title(f"(a) CSR stores the same TF-IDF in {r20.dense_tfidf_vs_csr:.0f}x less memory", fontsize=11)
    ax.set_xticks(mem.n_docs, [f"{n // 1000}K" for n in mem.n_docs])
    ax.legend(loc="upper left")
    ax = axes[1]
    style = {"sparse.sparse": "sparse", "dense.dense": "numpy", "sparse.dense": "numba", "embedding 768-d": "embedding"}
    for combo in style:                                   # legend order passes the colour-blindness check
        for impl, g in times[times.combo == combo].groupby("impl", sort=False):
            ls = "-" if impl == "ours" or combo == "embedding 768-d" else "--"
            P.line(ax, g.n_docs, g.median_s * 1e3, style[combo], f"{combo} ({impl})", linestyle=ls,
                   alpha=1 if ls == "-" else 0.75)
    ax.set_xscale("log"); ax.set_yscale("log")
    ns = sorted(times.n_docs.unique())
    ax.set_xticks(ns, [f"{n // 1000}K" for n in ns])
    ax.set_xlabel("corpus size (documents)"); ax.set_ylabel("time for 200 queries (ms, log)")
    t = times[times.n_docs == ns[-1]].set_index(["combo", "impl"]).median_s
    ax.set_title(f"(b) Inverted-index sparse.sparse is {t[('dense.dense', 'numpy')] / t[('sparse.sparse', 'ours')]:.0f}x "
                 f"faster than dense.dense (1 thread)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    P.savefig(fig, "fig2_sparse_vs_dense", STAGE)
    return fig


def density_sweep(n_docs=5_000, vocab=10_000, n_queries=200):
    """Random sparse matrices of growing density: where does dense start to win?"""
    densities = [1e-3, 1e-2, 0.1] if config.QUICK else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3]
    numba.set_num_threads(1)
    rows = []
    with bench.blas_threads(1):
        for d in densities:
            Xr = sd.random_csr(n_docs, vocab, d, seed=1)
            Qr = sd.random_csr(n_queries, vocab, d, seed=2)
            inv = sd.build_inverted_index(Xr)
            Xrd, Qrd = Xr.toarray(), Qr.toarray()
            rows.append({"density": d,
                         "sparse_ours_s": bench.benchmark(sd.sparse_dot_sparse, Qr, inv, n_docs, warmup=1, runs=5)["median"],
                         "dense_blas_s": bench.benchmark(sd.ref_dense_dense, Qrd, Xrd, warmup=1, runs=5)["median"],
                         "csr_mb": sd.memory_mb(Xr), "dense_mb": sd.memory_mb(Xrd)})
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    dens = pd.DataFrame(rows)
    bench.save_csv(dens, "density_sweep", STAGE)
    lose = dens[dens.sparse_ours_s > dens.dense_blas_s]
    cross = float(lose.density.min()) if len(lose) else None
    bench.update_summary(STAGE, density_crossover=cross)
    return dens, cross


def plot_density(dens, cross, our_density):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    P.line(ax, dens.density, dens.sparse_ours_s * 1e3, "sparse", "sparse.sparse (ours, inverted index)")
    P.line(ax, dens.density, dens.dense_blas_s * 1e3, "numpy", "dense.dense (BLAS)")
    ax.axvline(our_density, color=P.MUTED, linewidth=1)
    ax.annotate(f"our TF-IDF\n({our_density:.2%})", xy=(our_density, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -28), textcoords="offset points", fontsize=9, color=P.INK_2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("density (fraction of non-zeros, log)"); ax.set_ylabel("time (ms, log)")
    ax.set_title(f"Dense wins above ~{cross:.0%} density" if cross else "Sparse wins over the whole range", fontsize=11)
    ax.legend(loc="upper left")
    P.savefig(fig, "fig2b_density_crossover", STAGE)
    return fig


def hit_rates(S, targets):
    tgt = S[np.arange(len(targets)), targets][:, None]
    rank = (S > tgt).sum(axis=1)
    return float((rank < 1).mean()), float((rank < 10).mean())


def quality(queries, X_tfidf, Q_tfidf, X_emb, Q_emb):
    targets = queries.target_doc.to_numpy()
    h_t = hit_rates(sd.ref_sparse_sparse(Q_tfidf, X_tfidf), targets)
    h_e = hit_rates(Q_emb @ X_emb.T, targets)
    q = pd.DataFrame({"hit@1": [h_t[0], h_e[0]], "hit@10": [h_t[1], h_e[1]]},
                     index=["TF-IDF (sparse)", "Embedding (dense, 768-d)"])
    bench.update_summary(STAGE, quality=q.to_dict())
    return q


PARAPHRASES = ["speeding up text generation of large language models",
               "making neural networks small enough to run on a phone"]


def paraphrase_examples(corpus, vec, X_tfidf, X_emb, model, queries=PARAPHRASES, top=3):
    """Queries that share few keywords with the abstracts: what does each representation return?"""
    out = []
    for text in queries:
        q_t = vec.transform([text]).astype(np.float32)
        q_e = data.normalize_rows(data.embed_texts([text], model, show_progress=False))
        out.append({"query": text, "tfidf_terms": vec.inverse_transform(q_t)[0].tolist(),
                    "tfidf": [corpus.title[i] for i in np.argsort(-(q_t @ X_tfidf.T).toarray()[0])[:top]],
                    "embedding": [corpus.title[i] for i in np.argsort(-(q_e @ X_emb.T)[0])[:top]]})
    bench.update_summary(STAGE, paraphrase_examples=out)
    return out


def finalize(mem, times, q, cross, stats):
    r = mem.iloc[-1]
    t = times[times.n_docs == times.n_docs.max()].set_index(["combo", "impl"]).median_s
    bench.update_summary(STAGE, memory_20k=r.to_dict(), times_ms_largest_n={f"{c} | {i}": v * 1e3 for (c, i), v in t.items()})
    h_t, h_e = q.loc["TF-IDF (sparse)"], q.loc["Embedding (dense, 768-d)"]
    cap = (f"Figure 2: For the same TF-IDF numbers, CSR storage needs {r.dense_tfidf_vs_csr:.0f}x less memory than a dense "
           f"array ({r.tfidf_csr_mb:.0f} MB vs {r.tfidf_dense_mb:.0f} MB at 20K docs) and our inverted-index sparse-sparse "
           f"kernel is {t[('dense.dense', 'numpy')] / t[('sparse.sparse', 'ours')]:.0f}x faster than single-threaded BLAS "
           f"dense-dense and {t[('embedding 768-d', 'numpy')] / t[('sparse.sparse', 'ours')]:.0f}x faster than scoring 768-d "
           f"embeddings; we still choose dense embeddings ({r.embedding_dense_mb:.0f} MB) because they rank the correct "
           f"abstract first more often (hit@1 {h_e['hit@1']:.0%} vs {h_t['hit@1']:.0%}; hit@10 {h_e['hit@10']:.0%} vs "
           f"{h_t['hit@10']:.0%}) and match paraphrased queries that share no keywords with the documents.")
    bench.save_caption(STAGE, "fig2_sparse_vs_dense", cap)
    if cross:
        bench.save_caption(STAGE, "fig2b_density_crossover",
            f"Figure 2b: Sparse storage only pays off for very sparse data: with single-threaded kernels, BLAS dense-dense "
            f"overtakes our sparse kernel at about {cross:.0%} density, while our TF-IDF matrix is {stats['density']:.2%} dense.")
    return cap


def main():
    P.setup_style()
    corpus, queries, X_emb, Q_emb, vec, X_tfidf, Q_tfidf = load()
    stats = tfidf_stats(X_tfidf, Q_tfidf)
    mem = memory_table(X_tfidf, X_emb)
    print(check_kernels(X_tfidf, Q_tfidf))
    times = timing(X_tfidf, Q_tfidf, X_emb, Q_emb)
    plot_memory_time(mem, times)
    dens, cross = density_sweep()
    plot_density(dens, cross, stats["density"])
    q = quality(queries, X_tfidf, Q_tfidf, X_emb, Q_emb)
    print(q)
    paraphrase_examples(corpus, vec, X_tfidf, X_emb, data.load_model())
    print(finalize(mem, times, q, cross, stats))


if __name__ == "__main__":
    main()
