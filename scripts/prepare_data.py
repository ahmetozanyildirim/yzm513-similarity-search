"""Build the whole dataset from scratch (same steps as notebook 00, headless).

    python scripts/prepare_data.py            # download (resumable) + prepare + embed
    python scripts/prepare_data.py --skip-download

Outputs (data/processed/): corpus.jsonl, queries.jsonl,
corpus_embeddings.npy (20000 x 768 float32), query_embeddings.npy (200 x 768 float32)
"""

import argparse
import time

import numpy as np

from hpss import bench, config, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        data.fetch_arxiv()

    raw = data.load_raw()
    corpus, report = data.prepare_corpus(raw)
    queries = data.make_queries(corpus)
    data.save_jsonl(corpus, config.CORPUS_JSONL)
    data.save_jsonl(queries, config.QUERIES_JSONL)
    bench.update_summary("0_data", preprocessing=report)
    print("preprocessing:", report)

    if not args.skip_embed:
        model = data.load_model()
        t0 = time.perf_counter()
        X = data.embed_texts(corpus["abstract"], model)
        Q = data.embed_texts(queries["text"], model)
        print(f"embedding took {time.perf_counter() - t0:.0f} s (not part of any benchmark)")
        np.save(config.CORPUS_EMB, X)
        np.save(config.QUERY_EMB, Q)
        print("corpus", X.shape, "queries", Q.shape)


if __name__ == "__main__":
    main()
