"""Project-wide paths and constants. Every experiment imports from here,
so changing a number in one place changes it everywhere."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the repository root (this folder is what gets submitted)

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
# HPSS_RESULTS_DIR redirects all outputs (used for quick smoke tests so real results are not overwritten)
RESULTS = Path(os.environ.get("HPSS_RESULTS_DIR", ROOT / "results"))

# One results folder per project stage (= report section)
STAGES = {
    "0_data": "Dataset and environment (report section 1)",
    "1_baseline": "Phase 1 - baseline and profiling (section 2)",
    "2a_sparse_dense": "Phase 2A - sparse vs dense (section 3.1)",
    "2b_dim_reduction": "Phase 2B - dimensionality reduction (section 3.2)",
    "2c_vectorisation": "Phase 2C - vectorisation vs JIT (section 3.3)",
    "3_parallelism": "Phase 3 - parallelism and concurrency (section 4)",
    "4_final_system": "Phase 4 - final system and scaling (section 6)",
    "5_bonus": "Bonus - roofline, ANN, machine comparison (appendix)",
}

for _d in (DATA_RAW, DATA_PROC):
    _d.mkdir(parents=True, exist_ok=True)


def stage_dir(stage: str) -> Path:
    """results/<stage>/ - figures, raw CSV/JSON and results.json of one stage."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; choose from {list(STAGES)}")
    d = RESULTS / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- dataset -------------------------------------------------------------
ARXIV_CATEGORIES = ("cs.AI", "cs.CL", "cs.LG")
ARXIV_FIRST_MONTH = (2025, 1)        # inclusive
ARXIV_LAST_MONTH = (2026, 9)         # inclusive
N_CORPUS = 20_000                    # documents in the full corpus
N_QUERIES = 200                      # spec: 50-200 queries
MIN_WORDS = 50                       # spec: average length >= 50 tokens

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"   # 768-d
EMBED_DIM = 768

# ---- experiments ---------------------------------------------------------
SEED = 42
TOP_K = 10
CORPUS_SIZES = [1_000, 5_000, 10_000, 20_000]   # spec: scaling sizes
WARMUP = 3
RUNS = 5                                        # spec: >= 5 timed runs
TOL = 1e-5                                      # spec: correctness tolerance

# HPSS_QUICK=1 shrinks the slow experiments (smoke test only - never for reported numbers)
QUICK = bool(os.environ.get("HPSS_QUICK"))

# File names (single source of truth)
RAW_JSONL = DATA_RAW / "arxiv_2025_2026.jsonl"
CORPUS_JSONL = DATA_PROC / "corpus.jsonl"
QUERIES_JSONL = DATA_PROC / "queries.jsonl"
CORPUS_EMB = DATA_PROC / "corpus_embeddings.npy"
QUERY_EMB = DATA_PROC / "query_embeddings.npy"


def baseline_topk_path() -> Path:
    """Reference top-k of the naive baseline (written in stage 1, read by every later stage)."""
    return stage_dir("1_baseline") / "baseline_topk.npz"
