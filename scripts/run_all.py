"""Reproduce every figure and number of the report, stage by stage.

    python scripts/run_all.py              # all stages (~25 min on an 8-core laptop)
    python scripts/run_all.py 1 2a         # only these stages

Each stage writes into results/<stage>/. Close other CPU-heavy programs first:
benchmarks are sensitive to background load.
"""

import importlib
import sys
import time

MODULES = {"0": "exp0_data", "1": "exp1_baseline", "2a": "exp2a_sparse_dense", "2b": "exp2b_dim_reduction",
           "2c": "exp2c_vectorisation", "3": "exp3_parallelism", "4": "exp4_final_system", "5": "exp5_bonus"}


def main():
    wanted = sys.argv[1:] or list(MODULES)
    for key in wanted:
        t0 = time.perf_counter()
        print(f"=== stage {key}: {MODULES[key]} ===", flush=True)
        importlib.import_module(f"hpss.experiments.{MODULES[key]}").main()
        print(f"=== stage {key} done in {time.perf_counter() - t0:.0f} s ===\n", flush=True)


if __name__ == "__main__":
    main()
