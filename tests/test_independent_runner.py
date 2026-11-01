"""Execute the independent experiment entry point, not a mocked timing loop."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_independent_runner_smoke_records_real_timings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts/validate_independent.py"
    assert runner.exists(), "Independent validation runner has not been implemented"
    output = tmp_path / "validation"
    env = dict(os.environ, NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(runner), "--smoke", "--sizes", "32",
                             "--queries", "8", "--dimensions", "4", "--features", "128",
                             "--output", str(output)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run_kind"] == "smoke"
    assert summary["all_exact_checks_passed"] is True
    assert summary["representation"]["dimensions"] == 4
    assert len(summary["rows"]) >= 8
    for row in summary["rows"]:
        assert row["warmup"] >= 3 and row["runs"] >= 5
        assert len(row["raw_times"]) == row["runs"]
        assert row["all_ok"] is True
    assert (output / "timings.csv").exists()
    assert (output / "scaling.png").exists()
    assert len(summary["fingerprints"]["corpus_jsonl_sha256"]) == 64
