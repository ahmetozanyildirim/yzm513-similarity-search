"""One module per project stage. Each can be run on its own and writes its
figures, raw data and results.json into results/<stage>/:

    python -m hpss.experiments.exp0_data           # dataset statistics       (report section 1)
    python -m hpss.experiments.exp1_baseline       # Phase 1                  (section 2)
    python -m hpss.experiments.exp2a_sparse_dense  # Phase 2A                 (section 3.1)
    python -m hpss.experiments.exp2b_dim_reduction # Phase 2B                 (section 3.2)
    python -m hpss.experiments.exp2c_vectorisation # Phase 2C                 (section 3.3)
    python -m hpss.experiments.exp3_parallelism    # Phase 3                  (section 4)
    python -m hpss.experiments.exp4_final_system   # Phase 4                  (section 6)
    python -m hpss.experiments.exp5_bonus          # roofline, ANN            (appendix)

Stages must run in this order (later stages verify against the baseline of stage 1).
"""
