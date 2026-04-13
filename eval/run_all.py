"""
eval/run_all.py — Benchmark orchestrator.

Runs one or more harnesses against one or more datasets and writes all
results to a timestamped directory under eval/runs/.

Usage:
    python eval/run_all.py                                # all harnesses, all datasets
    python eval/run_all.py --harness router               # router only
    python eval/run_all.py --harness retrieval --dataset courses --limit 50
    python eval/run_all.py --harness e2e --judge          # with LLM scoring
    python eval/run_all.py --harness router retrieval e2e --dataset courses hard

Available harnesses:
    router      collection-routing accuracy (no Chroma needed)
    retrieval   recall@k / MRR
    e2e         generation quality (citation + field match + optional LLM judge)
    multiturn   polarity / pronoun resolution (pytest-based)
    file        file-upload pipeline
    perf        latency + cost

Available dataset aliases:
    courses   eval/datasets/courses.jsonl
    majors    eval/datasets/majors.jsonl
    policies  eval/datasets/policies.jsonl
    hard      eval/datasets/hard_cases.jsonl
    all       all four of the above
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so that both `rag_chatbot` and `eval` are importable
# regardless of where this script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before any harness module is imported (harnesses do this too, but
# lazy imports inside functions run after the top-level load_dotenv call here).
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_PROJECT_ROOT / ".env")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────────────────────────────────────

_DATASET_MAP = {
    "courses":  Path("eval/datasets/courses.jsonl"),
    "majors":   Path("eval/datasets/majors.jsonl"),
    "policies": Path("eval/datasets/policies.jsonl"),
    "hard":     Path("eval/datasets/hard_cases.jsonl"),
}

_ALL_DATASETS = list(_DATASET_MAP.values())


def _resolve_datasets(names: list[str]) -> list[Path]:
    if "all" in names:
        return _ALL_DATASETS
    paths = []
    for name in names:
        if name in _DATASET_MAP:
            paths.append(_DATASET_MAP[name])
        else:
            # Treat as a direct file path
            paths.append(Path(name))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Per-harness runners
# ──────────────────────────────────────────────────────────────────────────────

def _load_jsonl(paths: list[Path], limit: int | None) -> list[dict]:
    examples = []
    for p in paths:
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]
    return examples


def _run_router(datasets: list[Path], limit: int | None, out_dir: Path, verbose: bool, **_) -> dict:
    from eval.harnesses.router_eval import run, print_summary
    examples = _load_jsonl(datasets, limit)
    if not examples:
        return {}
    print(f"\n[router] {len(examples)} examples")
    result = asyncio.run(run(examples, verbose=verbose))
    print_summary(result["summary"])
    out = out_dir / "router_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  ->{out}")
    return result["summary"]


def _run_retrieval(datasets: list[Path], limit: int | None, out_dir: Path, verbose: bool, **_) -> dict:
    from eval.harnesses.retrieval_eval import run, print_summary
    examples = [e for e in _load_jsonl(datasets, limit) if e.get("expected_code") or e.get("golden_url")]
    if not examples:
        return {}
    print(f"\n[retrieval] {len(examples)} examples with golden signals")
    result = asyncio.run(run(examples, verbose=verbose))
    print_summary(result["summary"])
    out = out_dir / "retrieval_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  ->{out}")
    return result["summary"]


def _run_e2e(datasets: list[Path], limit: int | None, out_dir: Path, verbose: bool, judge: bool, judge_model: str, **_) -> dict:
    from eval.harnesses.e2e_eval import run, print_summary
    examples = [e for e in _load_jsonl(datasets, limit) if e.get("type") != "multiturn"]
    if not examples:
        return {}
    print(f"\n[e2e] {len(examples)} examples (judge={'on' if judge else 'off'})")
    result = asyncio.run(run(examples, use_judge=judge, judge_model=judge_model, verbose=verbose))
    print_summary(result["summary"])

    out_jsonl = out_dir / "e2e_results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in result["results"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_summary = out_dir / "e2e_summary.json"
    out_summary.write_text(json.dumps(result["summary"], indent=2))
    print(f"  ->{out_jsonl}")
    return result["summary"]


def _run_multiturn(out_dir: Path, **_) -> dict:
    """Delegate to pytest for clean pass/fail reporting."""
    print("\n[multiturn] Running via pytest...")
    cmd = [sys.executable, "-m", "pytest", "eval/harnesses/multiturn_eval.py", "-v",
           "--tb=short", f"--junit-xml={out_dir}/multiturn_junit.xml"]
    result = subprocess.run(cmd, capture_output=False)
    return {"pytest_exit_code": result.returncode}


def _run_file(out_dir: Path, verbose: bool, **_) -> dict:
    from eval.harnesses.file_eval import run, print_summary
    print("\n[file] Running file-upload harness...")
    result = asyncio.run(run(verbose=verbose))
    print_summary(result["summary"])
    out = out_dir / "file_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  ->{out}")
    return result["summary"]


def _run_perf(datasets: list[Path], limit: int | None, out_dir: Path, verbose: bool, **_) -> dict:
    from eval.harnesses.perf_eval import run, print_summary
    examples = _load_jsonl(datasets, limit or 20)
    if not examples:
        return {}
    print(f"\n[perf] {len(examples)} examples")
    result = asyncio.run(run(examples, verbose=verbose))
    print_summary(result["summary"])

    import csv
    ok = [r for r in result["results"] if "error" not in r]
    if ok:
        csv_path = out_dir / "perf.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ok[0].keys())
            writer.writeheader()
            writer.writerows(ok)
        print(f"  ->{csv_path}")

    out = out_dir / "perf_summary.json"
    out.write_text(json.dumps(result["summary"], indent=2))
    return result["summary"]


_HARNESS_FNS = {
    "router":    _run_router,
    "retrieval": _run_retrieval,
    "e2e":       _run_e2e,
    "multiturn": _run_multiturn,
    "file":      _run_file,
    "perf":      _run_perf,
}

_ALL_HARNESSES = list(_HARNESS_FNS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UCI RAG chatbot benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--harness", nargs="+", default=["all"],
        choices=_ALL_HARNESSES + ["all"],
        help="Harnesses to run (default: all)",
    )
    parser.add_argument(
        "--dataset", nargs="+", default=["all"],
        help="Dataset aliases or paths (courses|majors|policies|hard|all)",
    )
    parser.add_argument("--limit",       type=int,   default=None,         help="Max examples per harness")
    parser.add_argument("--judge",       action="store_true",               help="Enable LLM judge in e2e harness")
    parser.add_argument("--judge-model", default="gpt-5.4-mini",           help="Model for LLM judge")
    parser.add_argument("--verbose",     action="store_true",               help="Per-example output")
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: eval/runs/<timestamp>)",
    )
    args = parser.parse_args()

    harnesses = _ALL_HARNESSES if "all" in args.harness else args.harness
    datasets  = _resolve_datasets(args.dataset)
    out_dir   = Path(args.out) if args.out else Path(f"eval/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBenchmark run -> {out_dir}")
    print(f"Harnesses : {harnesses}")
    print(f"Datasets  : {[str(d) for d in datasets]}")
    if args.limit:
        print(f"Limit     : {args.limit} examples per harness")

    kwargs = dict(
        datasets=datasets,
        limit=args.limit,
        out_dir=out_dir,
        verbose=args.verbose,
        judge=args.judge,
        judge_model=args.judge_model,
    )

    all_summaries = {}
    for name in harnesses:
        try:
            summary = _HARNESS_FNS[name](**kwargs)
            all_summaries[name] = summary
        except Exception as exc:
            print(f"\n[{name}] ERROR: {exc}")
            all_summaries[name] = {"error": str(exc)}

    # Write combined summary
    combined = out_dir / "summary.json"
    combined.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    print(f"\nCombined summary -> {combined}")
    print("Run 'python eval/report.py' to compare against the previous run.\n")


if __name__ == "__main__":
    main()
