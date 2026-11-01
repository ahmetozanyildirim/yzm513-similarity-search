# Results overview (auto-generated - rerun `python scripts/collect_results.py`)

## 0_data - Dataset and environment (report section 1)
| key | value |
|---|---|
| system | cpu: AMD Ryzen 7 9800X3D 8-Core Processor, physical_cores: 8, logical_cores: 16, ram_gb: 31.2, os: Windows 11 (10.0.26200), python: 3.12.10, numpy: 2.5.4, scipy: 1.18.1, scikit-learn: 1.9.1, numba: 0.68.0, blas: [2 items - see results.json], numba_threads: 16, in_colab: False |
| preprocessing | raw_records: 31500, after_dedup_id: 24866, after_dedup_text: 24865, after_min_words: 24833, final_docs: 20000, raw_cells: 63 |
| dataset | n_docs: 20000, n_queries: 200, words_mean: 178, words_median: 177, tokens_mean: 261.1, tokens_median: 258, tokens_min: 59, tokens_max: 1052, frac_truncated: 0.0382, model_max_tokens: 384, query_words_mean: 10.4, date_min: 2025-01-01, date_max: 2026-09-05, primary_category_counts: cs.CL: 5869, cs.LG: 5212, cs.AI: 2303, cs.CV: 1786, cs.CR: 515, cs.RO: 416, stat.ML: 381, cs.IR: 338 |
| sanity | hit1: 0.74, hit10: 0.9, emb_norm_min: 1, emb_norm_max: 1 |
| dataset_paragraph | We use 20,000 abstracts of arXiv papers submitted between 2025-01-01 and 2026-09-05 in cs.AI, cs.CL and cs.LG (Option B). Raw API records: 31,500; after removing cross-listed duplicates (24,866), identical texts (24,865) and abstracts shorter than 50 words (24,833), we randomly sample 20,000 documents (seed 42). Abstracts have 261 tokens on average (median 258; 3.8% exceed the model's 384-token window). Queries are the titles of 200 papers (10.4 words on average); the matching abstract is the known correct answer: with all-mpnet-base-v2 (768-d) embeddings it is ranked first for 74% of the queries and within the top-10 for 90%. |

![fig0_dataset](../results/0_data/fig0_dataset.png)

> Figure 0: The corpus consists of 20,000 arXiv abstracts (2025-26) with 261 tokens on average, well above the 50-token minimum; only 3.8% exceed the embedding model's 384-token window.

## 1_baseline - Phase 1 - baseline and profiling (section 2)
| key | value |
|---|---|
| baseline | n_docs: 20000, queries: 200, median_s: 154.4, mean_s: 154.4, std_s: 0.07344, min_s: 154.3, max_s: 154.5, per_query_s: 0.7718, per_pair_us: 38.59 |
| naive_by_n | 1000: 7.727, 5000: 38.62, 10000: 77.59, 20000: 154.4 |
| cprofile | profiled_s: 190.1, plain_s: 154.4, overhead_x: 1.231 |
| reprofile_top_lines | [3 items - see results.json] |
| memory_layout | python_vector_kb: 24.05, numpy_vector_kb: 3, ratio: 8.018, python_corpus_mb: 469.8, numpy_corpus_mb: 58.59 |
| cprofile_cosine_pct | 98.44 |
| cprofile_sort_pct | 0.4127 |
| line_top4_pct | 99.79 |

![fig1_profiling](../results/1_baseline/fig1_profiling.png)

> Figure 1: The naive search spends 98% of its time inside cosine_similarity_python (called once per query-document pair) and only 0.4% sorting; inside the function, the loop header and the three multiply-add lines take 99.8% in near-equal shares - the cost is interpreter overhead per float operation, and two of the three multiply-adds only recompute norms.

## 2a_sparse_dense - Phase 2A - sparse vs dense (section 3.1)
| key | value |
|---|---|
| tfidf | vocab: 10000, nnz: 1832911, density: 0.009165, nnz_per_doc: 91.65, nnz_per_query: 8.045 |
| density_crossover | 0.2 |
| quality | hit@1: TF-IDF (sparse): 0.685, Embedding (dense, 768-d): 0.74, hit@10: TF-IDF (sparse): 0.9, Embedding (dense, 768-d): 0.9 |
| memory_20k | n_docs: 2e+04, tfidf_csr_mb: 14.06, tfidf_dense_mb: 762.9, embedding_dense_mb: 58.59, dense_tfidf_vs_csr: 54.26 |
| times_ms_largest_n | sparse.sparse | ours: 3.672, sparse.sparse | scipy: 26.45, sparse.dense | ours: 25.64, sparse.dense | scipy: 53.46, dense.dense | ours: 1851, dense.dense | numpy: 295.6, embedding 768-d | numpy: 23.22 |

![fig2_sparse_vs_dense](../results/2a_sparse_dense/fig2_sparse_vs_dense.png)

> Figure 2: For the same TF-IDF numbers, CSR storage needs 54x less memory than a dense array (14 MB vs 763 MB at 20K docs) and our inverted-index sparse-sparse kernel is 81x faster than single-threaded BLAS dense-dense and 6x faster than scoring 768-d embeddings; we still choose dense embeddings (59 MB) because they rank the correct abstract first more often (hit@1 74% vs 68%; hit@10 90% vs 90%) and match paraphrased queries that share no keywords with the documents.

![fig2b_density_crossover](../results/2a_sparse_dense/fig2b_density_crossover.png)

> Figure 2b: Sparse storage only pays off for very sparse data: with single-threaded kernels, BLAS dense-dense overtakes our sparse kernel at about 20% density, while our TF-IDF matrix is 0.92% dense.

## 2b_dim_reduction - Phase 2B - dimensionality reduction (section 3.2)
| key | value |
|---|---|
| power_iteration_check | k: 50, iterations: 100, sv_rel_err_median: 0.0001098, sv_rel_err_max: 0.006065, subspace_overlap_min: 0.9933, recon_error_ours: 74.71, recon_error_optimal: 74.71, excess_pct: 0.0001325 |
| full_svd | median_s: 0.5539, min_s: 0.5273, max_s: 0.7932 |
| k_crossover_truncated_vs_full | 300 |
| mean_direction | mean_norm: 0.5231, cos_v1_mean: 0.9976, sigma1_energy: 0.2914 |
| chosen | method: truncated_svd, k: 128, n_candidates: 100 |
| time_k100 | numpy_svd: 0.5539, pca: 0.1674, power_iteration: 1.844, truncated_svd: 0.247 |
| rel_error | power_iteration: 0.4204, numpy_svd: 0.4204, truncated_svd: 0.4212, pca: 0.4198, optimal (Eckart-Young): 0.4204 |
| recall_k128 | recall_plain: 0.795, recall_rerank100: 1, search_plain_ms: 13.45, search_rerank_ms: 22.04, exact_768_ms: 15.39 |
| pca_solver | covariance_eigh |

![fig3_dim_reduction](../results/2b_dim_reduction/fig3_dim_reduction.png)

> Figure 3: Truncated (randomised) SVD reaches the optimal Eckart-Young error at k=100 (0.421 relative) in 0.25 s versus 0.55 s for the full SVD and 1.84 s for our power iteration; the full SVD overtakes it from k~300 on; at k=128, re-ranking 100 reduced-space candidates in full 768-d recovers 100.0% of the exact top-10 (only 80% without re-ranking), whereas PCA - fastest and lowest-error - reaches only 98.1% because centring changes the cosine geometry the search relies on.

## 2c_vectorisation - Phase 2C - vectorisation vs JIT (section 3.3)
| key | value |
|---|---|
| numba_compile | first_compile_in_process_s: 0.4579, compile_median_s: 0.1209, second_call_s: 0.001077 |
| numba_cache | first_call_from_cache_s: 0.2641, second_call_s: 0.001085 |
| main_size | queries: 200, docs: 20000 |
| memory_layout | [7 items - see results.json] |
| normalise_peak_mb | in_place: 58.62, new_array: 117.2 |
| gathered_candidates_mb | 58.59 |
| batch_200_largest_ms | numpy: 26.46, numpy_1thread: 43.72, numba_parallel: 107.1, numba_fastmath: 20.19 |
| single_query_largest_ms | numpy: 20.09, numpy_1thread: 20.46, numba: 14.24, numba_parallel: 1.141, numba_fastmath: 0.2974 |

![fig4d_memory_layout](../results/2c_vectorisation/fig4d_memory_layout.png)

> Figure 4d: How the data is laid out and touched matters as much as the arithmetic: a fancy-index copy is 3,012x slower than a slice view, an einsum that bypasses BLAS 75x slower than Q @ X.T, gathering candidate rows before re-ranking 37x slower than reading them in place, and float64 2.1x slower than float32; row-major vs column-major access costs only 1.0x here because the 59 MB corpus fits in the 96 MB L3 cache.

![fig4_vectorisation](../results/2c_vectorisation/fig4_vectorisation.png)

> Figure 4: Compiling the same loops with Numba gives 46x over pure Python, prange 623x and prange+fastmath (SIMD) 3,465x, versus 2,530x for NumPy/BLAS. NumPy loses whenever its single-threaded normalisation temporaries dominate: for one query over 20,000 docs it needs 20.1 ms versus 0.30 ms for the fused Numba kernel; for a 200-query batch Numba fastmath is faster (Numba 20.2 ms vs NumPy 26.5 ms).

## 3_parallelism - Phase 3 - parallelism and concurrency (section 4)
| key | value |
|---|---|
| gil_sequential_s | 1.915 |
| gil_workload | queries: 48, docs: 1000 |
| pickling | workers: 8, corpus_pickle_mb: 6.597, bad_s: 0.7303, good_warm_s: 0.2659, good_cold_s: 2.043, bad_total_pickled_mb: 316.6 |
| switch_interval_s | 0.005 |
| phase2_reference | median_s: 0.03324, correct: True |
| batching | gemv_loop_1thread_ms: 106.2, gemv_loop_allthreads_ms: 37.16, gemm_batch_1thread_ms: 33.48, gemm_batch_best_ms: 16.03 |
| blas_serial_share | [2 items - see results.json] |
| race_demo | [2 items - see results.json] |
| granularity_workers | 8 |
| asyncio_sleep_resolution | requested_ms: 20, median_ms: 29.51 |
| ray_init_s | 12.09 |
| df_vocabulary | 174660 |
| df_words | 3560445 |
| amdahl_p | BLAS threads (NumPy batch): 0.5538, Dask delayed shards (threads): 0.7905, Numba prange (fused): 0.9777, ProcessPool + shared memory: 0.7432, ThreadPool + NumPy (1 BLAS thread/worker): 0.7429, ProcessPool (pure Python): 0.9486, ThreadPool (pure Python): 2.028e-08 |
| naive_full_s | 154.4 |

![fig5c_granularity](../results/3_parallelism/fig5c_granularity.png)

> Figure 5c: Granularity is the master dial: with 8 workers the 200 queries run fastest as tasks of 25 queries (10.0 ms, effective serial fraction f = 0.20); one task with all queries leaves the other workers idle (33.5 ms), and tiny tasks drown in overhead - 2-query tasks (46.6 ms) are even slower than 1-query tasks (17.6 ms) because BLAS switches from a matrix-vector to a matrix-matrix product that first packs the whole corpus.

![fig5d_dask_timeline](../results/3_parallelism/fig5d_dask_timeline.png)

> Figure 5d: Dask's task stream for one sharded search: the 8 shard tasks run side by side on 8 threads (6.7 ms each) and the single merge - the graph's narrow waist - takes 0.26 ms, only 4% of the 7.5 ms wall time, so a flat merge is enough at this scale.

![fig5_parallel_scaling](../results/3_parallelism/fig5_parallel_scaling.png)

> Figure 5: With pure Python, processes reach 8.5x (Amdahl p=0.95) while threads stay at 1.0x because of the GIL. For the vectorised search, BLAS threads stop at 2.1x (p=0.55) because the top-k selection after the matmul is single-threaded, whereas strategies that split the work into query chunks or shards also parallelise the selection (ProcessPool+shared memory p=0.74) and the fused Numba kernel has no serial part at all (p=0.98).

## 4_final_system - Phase 4 - final system and scaling (section 6)
| key | value |
|---|---|
| n_candidates | 100 |
| final_name | SVD-128 + re-rank(100) + parallel top-k |
| final_config | n_components: 128, n_candidates: 100, reducer: truncated_svd, topk: numba |
| correctness_all_ok | True |
| scale_inputs | scan_gbps: 58.88, ram_gb: 33.46, docs_per_s_measured: 9.095, assumed_serial_fraction_f: 0.02, assumed_docs_per_s: 1000, assumed_mtbf_days: 1000 |
| final_batch_ms | 3.889 |
| final_latency_ms | 0.2102 |

![fig6a_final_stage_profile](../results/4_final_system/fig6a_final_stage_profile.png)

> Figure 6a: Profiling our own engine shows that single-threaded argpartition takes 66% of the exact batched search (10.23 of 15.55 ms) - the serial fraction of Amdahl's law; a parallel Numba top-k cuts it to 5.36 ms and the final SVD-128 + re-rank(100) + parallel top-k engine needs 3.67 ms. Without OMP_WAIT_POLICY=PASSIVE, Numba's spinning OpenMP threads slow the following BLAS matmul from 4.4 to 34.1 ms (two thread pools competing for the same cores).

![fig6_scaling](../results/4_final_system/fig6_scaling.png)

> Figure 6: Search time grows at most linearly with corpus size (log-log slope 0.54 for the final engine between 5K and 20K and 0.85 up to 200K synthetic documents - below 1 because fixed per-call costs still matter), while the fused Numba kernel, which re-reads the corpus for every query, slows down x36 for 10x data once the corpus leaves the 96 MB L3 cache; the final engine answers 200 queries over 20K documents in 3.89 ms and a single query in 0.21 ms.

![fig7_all_versions](../results/4_final_system/fig7_all_versions.png)

> Figure 7: From 154 s (naive Python) to 3.89 ms (Final: SVD-128 + re-rank(100) + parallel top-k) on the same 200x20K workload, a 39,689x speed-up; Python-level fixes alone reach at most 4x - the big jump comes from leaving the interpreter (BLAS / Numba).

## 5_bonus - Bonus - roofline, ANN, machine comparison (appendix)
| key | value |
|---|---|
| roofline | peak_gflops: 2138, peak_gflops_1thread: 312.2, bw_dram_gbs: 45.33, bw_l3_gbs: 464 |
| hnsw_build_s | 1.671 |

![fig8_roofline](../results/5_bonus/fig8_roofline.png)

> Figure 8 (bonus): Batched GEMM search runs at 1109 GFLOP/s (AI~79 FLOP/B, 52% of the measured 2138 GFLOP/s compute roof) while per-query GEMV reaches only 204 GFLOP/s at AI=0.5, bounded by memory bandwidth (464 GB/s L3, 45 GB/s DRAM): batching queries is what moves the search from memory-bound to compute-bound.

![fig9_ann_tradeoff](../results/5_bonus/fig9_ann_tradeoff.png)

> Figure 9 (bonus): HNSW reaches recall@10 0.998 at 0.235 ms per query vs 0.537 ms for exact single-threaded search; our SimHash LSH needs to scan a large fraction of the corpus for high recall in 768 dimensions. At 20K documents exact search is already sub-millisecond, so ANN only pays off at much larger N.
