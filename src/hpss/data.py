"""Dataset: recent arXiv abstracts (cs.AI / cs.CL / cs.LG, Jan 2025 - Sep 2026).

Pipeline
  1. fetch_arxiv()     - download metadata through the public arXiv API (cached, resumable)
  2. prepare_corpus()  - clean, de-duplicate, filter short abstracts, shuffle, take N docs
  3. make_queries()    - queries = titles of papers; the paper's own abstract is the
                         "correct" answer, which gives a semantic sanity check
  4. embed_texts()     - dense 768-d embeddings with sentence-transformers (NOT benchmarked)
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
API_URL = "https://export.arxiv.org/api/query"


# ======================================================================
# 1. Download
# ======================================================================

def _months(first=config.ARXIV_FIRST_MONTH, last=config.ARXIV_LAST_MONTH):
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _month_range(y, m):
    nxt = (y + 1, 1) if m == 12 else (y, m + 1)
    start = f"{y:04d}{m:02d}010000"
    end_day = (pd.Timestamp(year=nxt[0], month=nxt[1], day=1) - pd.Timedelta(days=1)).day
    end = f"{y:04d}{m:02d}{end_day:02d}2359"
    return start, end


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _parse_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or 0)
    rows = []
    for e in root.findall(f"{ATOM}entry"):
        arxiv_id = e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        prim = e.find(f"{ARXIV_NS}primary_category")
        rows.append({
            "arxiv_id": arxiv_id,
            "title": clean_text(e.findtext(f"{ATOM}title")),
            "abstract": clean_text(e.findtext(f"{ATOM}summary")),
            "primary_category": prim.get("term") if prim is not None else "",
            "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
            "published": e.findtext(f"{ATOM}published", "")[:10],
        })
    return total, rows


def fetch_arxiv(per_cell=500, delay=3.2, out_path=config.RAW_JSONL, verbose=True):
    """Download `per_cell` papers for every (month, category) cell.

    The arXiv API asks for >= 3 s between calls, so ~63 cells take ~8-10 min.
    Already-downloaded cells are skipped, so an interrupted run can be resumed.
    """
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["cell"])
    cells = [(y, m, c) for (y, m) in _months() for c in config.ARXIV_CATEGORIES]
    todo = [cell for cell in cells if f"{cell[0]}-{cell[1]:02d}/{cell[2]}" not in done]
    if verbose:
        print(f"{len(cells)} cells total, {len(cells) - len(todo)} cached, {len(todo)} to download")

    with open(out_path, "a", encoding="utf-8") as out:
        for n, (y, m, cat) in enumerate(todo, 1):
            start, end = _month_range(y, m)
            query = f"cat:{cat} AND submittedDate:[{start} TO {end}]"
            params = urllib.parse.urlencode({
                "search_query": query, "start": 0, "max_results": per_cell,
                "sortBy": "submittedDate", "sortOrder": "ascending"})
            req = urllib.request.Request(f"{API_URL}?{params}",
                                         headers={"User-Agent": "yzm513-course-project/0.1"})
            rows = []
            for attempt in range(4):           # the API sometimes returns an empty page
                with urllib.request.urlopen(req, timeout=120) as r:
                    total, rows = _parse_feed(r.read())
                if rows or total == 0:
                    break
                time.sleep(delay * (attempt + 2))
            cell = f"{y}-{m:02d}/{cat}"
            for row in rows:
                row["cell"] = cell
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if verbose:
                print(f"[{n:>2}/{len(todo)}] {cell}: {len(rows)} papers")
            time.sleep(delay)
    return out_path


def load_raw(path=config.RAW_JSONL) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


# ======================================================================
# 2. Clean / filter / sample
# ======================================================================

def prepare_corpus(raw: pd.DataFrame, n_docs=config.N_CORPUS, min_words=config.MIN_WORDS,
                   seed=config.SEED):
    """Return (corpus_df, report) where report documents every preprocessing step."""
    report = {"raw_records": len(raw)}
    df = raw.drop_duplicates("arxiv_id")                 # cross-listed papers appear in several cells
    report["after_dedup_id"] = len(df)
    df = df.drop_duplicates("abstract")
    report["after_dedup_text"] = len(df)
    df = df.assign(n_words=df["abstract"].str.split().str.len())
    df = df[df["n_words"] >= min_words]
    report["after_min_words"] = len(df)
    if len(df) < n_docs:
        raise ValueError(f"only {len(df)} usable abstracts, need {n_docs}; download more")
    df = df.sample(n=n_docs, random_state=seed).reset_index(drop=True)
    df.insert(0, "doc_id", np.arange(len(df)))
    report["final_docs"] = len(df)
    cols = ["doc_id", "arxiv_id", "title", "abstract", "primary_category",
            "categories", "published", "n_words"]
    return df[cols], report


def make_queries(corpus: pd.DataFrame, n_queries=config.N_QUERIES, seed=config.SEED,
                 pool=config.CORPUS_SIZES[0]):
    """Queries are paper *titles*; the target is that paper's abstract.

    Targets are drawn from the first `pool` documents so that the correct
    answer exists at every corpus size in the scaling test (1K ... 20K).
    """
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(min(pool, len(corpus)), size=n_queries, replace=False))
    return pd.DataFrame({
        "query_id": np.arange(n_queries),
        "text": corpus.loc[targets, "title"].to_numpy(),
        "target_doc": targets,
    })


def save_jsonl(df: pd.DataFrame, path):
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def load_corpus(n=None) -> pd.DataFrame:
    df = pd.read_json(config.CORPUS_JSONL, lines=True)
    return df if n is None else df.iloc[:n]


def load_queries() -> pd.DataFrame:
    return pd.read_json(config.QUERIES_JSONL, lines=True)


# ======================================================================
# 3. Embeddings (pre-computed once; generation time is not benchmarked)
# ======================================================================

def load_model(name=config.EMBED_MODEL, device=None):
    """Embedding model; uses a GPU when one is available (e.g. Colab T4: ~1-2 min
    for the whole corpus instead of ~40 min on a laptop CPU)."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(name, device=device)


def embed_texts(texts, model=None, batch_size=64, show_progress=True):
    model = model or load_model()
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                       convert_to_numpy=True, normalize_embeddings=False)
    return emb.astype(np.float32)


def load_embeddings(n=None):
    """(corpus_emb [N x 768], query_emb [Q x 768]) as float32 arrays."""
    X = np.load(config.CORPUS_EMB)
    Q = np.load(config.QUERY_EMB)
    return (X if n is None else X[:n]), Q


def normalize_rows(A):
    """Unit rows; a zero vector stays zero, matching the Python cosine baseline."""
    A = np.asarray(A)
    if A.ndim != 2 or not A.shape[0] or not A.shape[1]:
        raise ValueError("expected a non-empty 2D matrix")
    if A.dtype.kind not in "fiu" or not np.isfinite(A).all():
        raise ValueError("matrix values must be finite real numbers")
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("row norms must be finite")
    dtype = np.result_type(A.dtype, np.float32)
    return np.divide(A, norms, out=np.zeros(A.shape, dtype=dtype), where=norms > 0)
