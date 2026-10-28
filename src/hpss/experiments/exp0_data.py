"""Stage 0 - environment, dataset statistics and a semantic sanity check (report section 1)."""

import json

import matplotlib.pyplot as plt
import numpy as np

from .. import bench, config, data
from .. import plotting as P

STAGE = "0_data"


def environment():
    info = bench.system_info()
    bench.update_summary(STAGE, system=info)
    return info


def load_or_prepare():
    """Corpus + queries + preprocessing report. Downloads / embeds only if the files are missing."""
    if not config.RAW_JSONL.exists():
        data.fetch_arxiv()                                    # ~10 min
    raw = data.load_raw()
    corpus, report = data.prepare_corpus(raw)
    queries = data.make_queries(corpus)
    if config.CORPUS_JSONL.exists() and data.load_corpus()["arxiv_id"].tolist() != corpus["arxiv_id"].tolist():
        config.CORPUS_EMB.unlink(missing_ok=True)             # raw data changed -> embeddings are stale
        config.QUERY_EMB.unlink(missing_ok=True)
    data.save_jsonl(corpus, config.CORPUS_JSONL)
    data.save_jsonl(queries, config.QUERIES_JSONL)
    report["raw_cells"] = int(raw["cell"].nunique())
    bench.update_summary(STAGE, preprocessing=report)
    return raw, corpus, queries, report


def statistics(corpus, queries, model):
    tok = model.tokenizer
    # Count the FULL length of every abstract (no truncation) to measure how many exceed the model window;
    # verbose=False silences the tokenizer's "sequence longer than 384" warning, which is expected here.
    n_tokens = np.array([len(ids) for ids in
                         tok(corpus["abstract"].tolist(), add_special_tokens=False, verbose=False)["input_ids"]])
    max_len = model.max_seq_length
    stats = {
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "words_mean": float(corpus["n_words"].mean()),
        "words_median": float(corpus["n_words"].median()),
        "tokens_mean": float(n_tokens.mean()),
        "tokens_median": float(np.median(n_tokens)),
        "tokens_min": int(n_tokens.min()),
        "tokens_max": int(n_tokens.max()),
        "frac_truncated": float((n_tokens > max_len - 2).mean()),
        "model_max_tokens": int(max_len),
        "query_words_mean": float(queries["text"].str.split().str.len().mean()),
        "date_min": str(corpus["published"].min())[:10],
        "date_max": str(corpus["published"].max())[:10],
        "primary_category_counts": corpus["primary_category"].value_counts().head(8).to_dict(),
    }
    bench.update_summary(STAGE, dataset=stats)
    return stats, n_tokens


def plot_dataset(corpus, stats, n_tokens):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    max_len = stats["model_max_tokens"]
    ax.hist(n_tokens, bins=60, color=P.color("numpy"), edgecolor="white", linewidth=0.6)
    ax.axvline(max_len, color=P.INK_2, linewidth=1)
    ax.annotate(f"model limit = {max_len} tokens\n({100 * stats['frac_truncated']:.1f}% truncated)",
                xy=(max_len, ax.get_ylim()[1] * 0.85), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=P.INK_2)
    ax.set_xlim(0, np.percentile(n_tokens, 99.9))
    ax.set_xlabel(f"abstract length (tokens; longest {n_tokens.max()}, axis cut at 99.9%)")
    ax.set_ylabel("documents")
    ax.set_title(f"Mean {stats['tokens_mean']:.0f} tokens (spec: >= 50)")
    ax = axes[1]
    top = corpus["primary_category"].value_counts().head(8)
    P.bars(ax, top.index.tolist(), top.values.tolist(), ["numpy"] * len(top), fmt="{:,.0f}", horizontal=True)
    ax.set_xlabel("documents")
    ax.set_title("Primary category (top 8)")
    P.savefig(fig, "fig0_dataset", STAGE)
    return fig


def load_or_embed(corpus, queries, model):
    """Pre-computed embeddings (generation time is not part of any benchmark)."""
    if config.CORPUS_EMB.exists() and config.QUERY_EMB.exists():
        X, Q = data.load_embeddings()
        if len(X) == len(corpus) and len(Q) == len(queries):
            return X, Q
    X = data.embed_texts(corpus["abstract"], model)           # ~40 min on a laptop CPU
    Q = data.embed_texts(queries["text"], model)
    np.save(config.CORPUS_EMB, X)
    np.save(config.QUERY_EMB, Q)
    return X, Q


def sanity_check(X, Q, queries):
    """Title -> own abstract: is the paper's abstract ranked 1st / within the top 10?"""
    Xn, Qn = data.normalize_rows(X), data.normalize_rows(Q)
    S = Qn @ Xn.T
    rank = (S > S[np.arange(len(Q)), queries["target_doc"]][:, None]).sum(axis=1)
    res = {"hit1": float((rank < 1).mean()), "hit10": float((rank < 10).mean()),
           "emb_norm_min": float(np.linalg.norm(X, axis=1).min()),
           "emb_norm_max": float(np.linalg.norm(X, axis=1).max())}
    bench.update_summary(STAGE, sanity=res)
    return res


def finalize(stats, report, sanity):
    s, r = stats, report
    paragraph = (
        f"We use {s['n_docs']:,} abstracts of arXiv papers submitted between {s['date_min']} and {s['date_max']} "
        f"in cs.AI, cs.CL and cs.LG (Option B). Raw API records: {r['raw_records']:,}; after removing cross-listed "
        f"duplicates ({r['after_dedup_id']:,}), identical texts ({r['after_dedup_text']:,}) and abstracts shorter "
        f"than 50 words ({r['after_min_words']:,}), we randomly sample {s['n_docs']:,} documents (seed 42). "
        f"Abstracts have {s['tokens_mean']:.0f} tokens on average (median {s['tokens_median']:.0f}; "
        f"{100 * s['frac_truncated']:.1f}% exceed the model's {s['model_max_tokens']}-token window). Queries are the "
        f"titles of {s['n_queries']} papers ({s['query_words_mean']:.1f} words on average); the matching abstract is "
        f"the known correct answer: with all-mpnet-base-v2 (768-d) embeddings it is ranked first for "
        f"{100 * sanity['hit1']:.0f}% of the queries and within the top-10 for {100 * sanity['hit10']:.0f}%.")
    bench.update_summary(STAGE, dataset_paragraph=paragraph)
    bench.save_caption(STAGE, "fig0_dataset",
        f"Figure 0: The corpus consists of {s['n_docs']:,} arXiv abstracts (2025-26) with {s['tokens_mean']:.0f} tokens "
        f"on average, well above the 50-token minimum; only {100 * s['frac_truncated']:.1f}% exceed the embedding "
        f"model's {s['model_max_tokens']}-token window.")
    return paragraph


def main():
    P.setup_style()
    print(json.dumps(environment(), indent=1))
    raw, corpus, queries, report = load_or_prepare()
    model = data.load_model()
    stats, n_tokens = statistics(corpus, queries, model)
    plot_dataset(corpus, stats, n_tokens)
    X, Q = load_or_embed(corpus, queries, model)
    sanity = sanity_check(X, Q, queries)
    print(finalize(stats, report, sanity))


if __name__ == "__main__":
    main()
