"""
eval/report.py — Benchmark regression reporter.

Reads the two most recent runs from eval/runs/ and prints a markdown diff
table so regressions are immediately visible.

Usage:
    python eval/report.py                   # diff latest vs previous
    python eval/report.py --run eval/runs/20260410_120000
    python eval/report.py --baseline eval/runs/20260401_090000 --run eval/runs/20260410_120000
    python eval/report.py --list            # list all available runs
"""

import argparse
import json
import sys
from pathlib import Path

_RUNS_DIR = Path("eval/runs")

# Metrics to show in the diff table, in display order.
# Format: (harness_key, metric_key, display_name, higher_is_better)
_METRICS = [
    ("router",    "collection_f1_mean",           "Router collection F1",      True),
    ("router",    "collection_exact_match",        "Router exact match",        True),
    ("router",    "requires_full_req_accuracy",    "Requires-full-req acc.",    True),
    ("router",    "major_kw_match_rate",           "Major kw match rate",       True),
    ("retrieval", "recall@1",                      "Retrieval Recall@1",        True),
    ("retrieval", "recall@3",                      "Retrieval Recall@3",        True),
    ("retrieval", "recall@10",                     "Retrieval Recall@10",       True),
    ("retrieval", "mrr",                           "Retrieval MRR",             True),
    ("retrieval", "direct_lookup_hit_rate",        "Direct lookup hit rate",    True),
    ("e2e",       "citation_ok_rate",              "E2E Citation accuracy",     True),
    ("e2e",       "field_match_rate",              "E2E Field match rate",      True),
    ("e2e",       "faithfulness_mean",             "E2E Faithfulness (1-5)",    True),
    ("e2e",       "relevance_mean",                "E2E Relevance (1-5)",       True),
    ("file",      "pass_rate",                     "File upload pass rate",     True),
    ("perf",      "latency_p50",                   "Perf latency p50 (s)",      False),
    ("perf",      "latency_p95",                   "Perf latency p95 (s)",      False),
    ("perf",      "cost_per_query_usd",            "Perf cost/query (USD)",     False),
]

_DELTA_THRESHOLD = 0.01  # changes smaller than this are treated as noise


def _list_runs() -> list[Path]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in _RUNS_DIR.iterdir() if p.is_dir() and (p / "summary.json").exists()],
        key=lambda p: p.name,
    )


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _get_metric(summary: dict, harness: str, key: str):
    harness_data = summary.get(harness, {})
    if isinstance(harness_data, dict):
        return harness_data.get(key)
    return None


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _delta_str(baseline_val, run_val, higher_is_better: bool) -> str:
    if baseline_val is None or run_val is None:
        return ""
    try:
        delta = float(run_val) - float(baseline_val)
    except (TypeError, ValueError):
        return ""
    if abs(delta) < _DELTA_THRESHOLD:
        return ""
    sign = "+" if delta > 0 else ""
    improved = (delta > 0) == higher_is_better
    arrow = "^" if improved else "v"
    return f"  {arrow} {sign}{delta:.4f}"


def print_report(baseline_dir: Path, run_dir: Path) -> None:
    baseline = _load_summary(baseline_dir)
    current  = _load_summary(run_dir)

    print(f"\n## Benchmark Comparison")
    print(f"**Baseline:** `{baseline_dir.name}`")
    print(f"**Current:**  `{run_dir.name}`\n")

    # Header
    col_w = [42, 12, 12, 18]
    header = f"{'Metric':<{col_w[0]}}  {'Baseline':>{col_w[1]}}  {'Current':>{col_w[2]}}  {'Delta':>{col_w[3]}}"
    sep    = "-" * len(header)
    print(header)
    print(sep)

    regressions = []
    for harness, key, label, higher_is_better in _METRICS:
        b_val = _get_metric(baseline, harness, key)
        c_val = _get_metric(current,  harness, key)

        delta = _delta_str(b_val, c_val, higher_is_better)

        # Flag regressions
        if b_val is not None and c_val is not None:
            try:
                d = float(c_val) - float(b_val)
                if abs(d) >= _DELTA_THRESHOLD:
                    improved = (d > 0) == higher_is_better
                    if not improved:
                        regressions.append((label, b_val, c_val, d))
            except (TypeError, ValueError):
                pass

        row = f"{label:<{col_w[0]}}  {_fmt(b_val):>{col_w[1]}}  {_fmt(c_val):>{col_w[2]}}  {delta}"
        print(row)

    print(sep)

    if regressions:
        print(f"\nWARNING: {len(regressions)} regression(s) detected:")
        for label, b, c, d in regressions:
            print(f"   {label}: {_fmt(b)} -> {_fmt(c)}  (delta {d:+.4f})")
    else:
        print("\nNo regressions detected.")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark regression reporter")
    parser.add_argument("--run",      default=None, help="Run directory to report (default: latest)")
    parser.add_argument("--baseline", default=None, help="Baseline run to compare against (default: second-latest)")
    parser.add_argument("--list",     action="store_true", help="List all available runs and exit")
    args = parser.parse_args()

    runs = _list_runs()

    if args.list:
        if not runs:
            print("No runs found in eval/runs/")
        else:
            for r in runs:
                print(f"  {r.name}")
        return

    if len(runs) == 0:
        print("No runs found. Run eval/run_all.py first.", file=sys.stderr)
        sys.exit(1)

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = runs[-1]

    if args.baseline:
        baseline_dir = Path(args.baseline)
    else:
        # Use second-to-last run as baseline; if only one run exists, report it alone
        if len(runs) < 2:
            print(f"Only one run available ({run_dir.name}). Run the benchmark again to enable comparison.\n")
            baseline = _load_summary(run_dir)
            print("Current summary:")
            print(json.dumps(baseline, indent=2))
            return
        baseline_dir = runs[-2] if run_dir == runs[-1] else runs[-1]

    if not run_dir.exists():
        print(f"ERROR: run directory '{run_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    if not baseline_dir.exists():
        print(f"ERROR: baseline directory '{baseline_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print_report(baseline_dir, run_dir)


if __name__ == "__main__":
    main()
