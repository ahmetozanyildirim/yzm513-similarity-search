"""Gather the key numbers and figure captions of every stage into report/results_overview.md
(the fact sheet the report is written from).

    python scripts/collect_results.py
"""

from pathlib import Path

from hpss import bench, config

OUT = config.ROOT / "report" / "results_overview.md"
SKIP = {"captions", "raw_times", "summary", "all_versions", "correctness", "stage_profile", "thread_contention",
        "python_fixes", "main", "reducer_search", "slopes", "paraphrase_examples"}


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items() if k not in SKIP)
    if isinstance(v, list):
        return f"[{len(v)} items - see results.json]"
    return str(v)


def main():
    parts = ["# Results overview (auto-generated - rerun `python scripts/collect_results.py`)\n"]
    for stage, title in config.STAGES.items():
        res = bench.load_summary(stage)
        if not res:
            parts += [f"## {stage} - {title}", "_not run yet_", ""]
            continue
        parts += [f"## {stage} - {title}", "| key | value |", "|---|---|"]
        parts += [f"| {k} | {fmt(v)} |" for k, v in res.items() if k not in SKIP]
        parts.append("")
        for fig_id, cap in res.get("captions", {}).items():
            rel = Path("..") / "results" / stage / f"{fig_id}.png"
            parts += [f"![{fig_id}]({rel.as_posix()})", "", f"> {cap}", ""]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
