"""Regression checks for correctness claims and reproducible cached inputs."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from hpss.bench import compare_topk
from hpss.data import normalize_rows
from hpss.search import SearchEngine, topk_from_scores


def test_normalize_rows_keeps_zero_vectors_finite():
    actual = normalize_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [[0.0, 0.0], [0.6, 0.8]])


def test_normalize_rows_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite|sonlu"):
        normalize_rows(np.array([[np.nan, 1.0]]))


def test_compare_topk_rejects_wrong_documents_with_copied_scores():
    check = compare_topk([[99, 98]], [[0.9, 0.8]], [[0, 1]], [[0.9, 0.8]])
    assert check["all_ok"] is False
    assert check["violations"] == 2


def test_compare_topk_allows_verified_tie_reordering():
    check = compare_topk([[1, 0]], [[0.9, 0.9]], [[0, 1]], [[0.9, 0.9]])
    assert check["all_ok"] is True
    assert check["tie_swaps"] == 2


def test_compare_topk_rejects_duplicate_documents():
    check = compare_topk([[0, 0]], [[0.9, 0.9]], [[0, 1]], [[0.9, 0.9]])
    assert check["all_ok"] is False


def test_compare_topk_rejects_broadcasted_shapes():
    with pytest.raises(ValueError, match="shape|boyut"):
        compare_topk([[0]], [[0.9]], [[0, 1]], [[0.9, 0.8]])


def test_topk_rejects_invalid_k():
    for k in (0, -1, 4, 1.5, True):
        with pytest.raises(ValueError, match="k|top"):
            topk_from_scores(np.array([[0.9, 0.8, 0.7]]), k)


def test_exact_engine_handles_zero_query_without_nan():
    engine = SearchEngine().fit(np.eye(3, dtype=np.float32))
    idx, scores = engine.search(np.zeros((1, 3)), k=2)
    assert idx.shape == scores.shape == (1, 2)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, [[0.0, 0.0]])


def test_cached_preparation_does_not_need_raw_download(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prepare_data_test_module", root / "scripts/prepare_data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pandas as pd
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    pd.DataFrame({"doc_id": [0], "abstract": ["cached document"], "title": ["cached title"]}).to_json(corpus_path, orient="records", lines=True)
    pd.DataFrame({"query_id": [0], "text": ["cached title"], "target_doc": [0]}).to_json(queries_path, orient="records", lines=True)
    before = corpus_path.read_bytes(), queries_path.read_bytes()
    monkeypatch.setattr(module.data, "load_corpus", lambda: pd.read_json(corpus_path, lines=True))
    monkeypatch.setattr(module.data, "load_queries", lambda: pd.read_json(queries_path, lines=True))
    monkeypatch.setattr(module.config, "CORPUS_JSONL", corpus_path)
    monkeypatch.setattr(module.config, "QUERIES_JSONL", queries_path)
    def no_download(*args, **kwargs):
        pytest.fail("Cached preparation must not fetch or read the missing raw download")
    monkeypatch.setattr(module.data, "fetch_arxiv", no_download)
    monkeypatch.setattr(module.data, "load_raw", no_download)
    monkeypatch.setattr(module.bench, "update_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--from-processed", "--skip-embed"])
    module.main()
    assert (corpus_path.read_bytes(), queries_path.read_bytes()) == before


@pytest.mark.parametrize("name", ["cosine_matrix_python", "cosine_matrix_numpy",
                                  "cosine_matrix_numba", "cosine_matrix_numba_parallel",
                                  "cosine_matrix_numba_fastmath"])
def test_every_cosine_kernel_matches_zero_vector_baseline(name):
    from hpss import kernels
    queries = np.array([[0.0, 0.0], [1.0, 0.0]])
    corpus = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    actual = getattr(kernels, name)(queries, corpus)
    np.testing.assert_allclose(actual, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_power_svd_rank_deficient_matrix_stays_finite():
    from hpss.svd import power_iteration_svd
    matrix = np.diag([3.0, 0.0])
    u, s, vt = power_iteration_svd(matrix, 2)
    assert np.isfinite(u).all()
    np.testing.assert_allclose((u * s) @ vt, matrix, atol=1e-10)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-10)


def test_power_svd_near_equal_singular_values_have_orthogonal_triplets():
    from hpss.svd import power_iteration_svd
    matrix = np.diag([1.0, 0.998])
    u, s, vt = power_iteration_svd(matrix, 2)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(matrix.T @ u, vt.T * s, atol=1e-10)
    np.testing.assert_allclose(s, [1.0, 0.998], atol=1e-10)


def test_numba_topk_rejects_more_results_than_documents():
    from hpss.kernels import topk_rows
    with pytest.raises(ValueError, match="k|top"):
        topk_rows(np.array([[0.1, 0.2]]), 3)


def test_power_svd_metadata_matches_returned_basis_at_iteration_limit():
    from hpss.svd import power_iteration_svd
    matrix = np.diag([3.0, 1.0])
    _, _, vt, meta = power_iteration_svd(matrix, 1, max_iter=1, tol=0.3, return_history=True)
    basis = vt.T
    product = matrix.T @ (matrix @ basis)
    expected = np.linalg.norm(product - basis @ (basis.T @ product)) / np.linalg.norm(product)
    assert meta["subspace_residual"] == pytest.approx(expected, abs=1e-12)
    assert meta["history"][-1] == pytest.approx(expected, abs=1e-12)
    assert meta["converged"] == bool(expected < 0.3)
