"""
eval/harnesses/retrieval_eval.py — Retrieval quality harness.

Calls retrieve() directly and measures whether the expected source document
ends up in the top-k returned chunks. Two golden signals per example:
  expected_code  — course code must appear in doc metadata["code"] at rank ≤ k
  golden_url     — source URL must appear in doc metadata["url"] at rank ≤ k

Metrics (computed at k = 1, 3, 10):
  recall@k        — fraction of examples where golden signal appears within top-k
  hit_rate@k      — alias for recall@k (same thing, different name in the literature)
  mrr             — mean reciprocal rank (1/rank of first hit, 0 if not found in top-10)
  direct_lookup   — fraction of course-code examples retrieved by direct lookup (rank 1)

Requires Chroma DBs (data/db/courses, data/db/policies, data/db/majors).
If they don't exist, the import of rag_chatbot.retriever will raise FileNotFoundError.

Usage:
    python eval/harnesses/retrieval_eval.py --dataset eval/datasets/courses.jsonl --limit 50
    python eval/harnesses/retrieval_eval.py --dataset eval/datasets/majors.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Chroma collections are loaded at import time — catch missing-DB errors clearly.
try:
    from rag_chatbot.retriever import retrieve
except FileNotFoundError as e:
    print(
        f"\nERROR: {e}\n"
        "Run ingest for all three collections before running retrieval eval:\n"
        "  python ingest/ingest.py --source data/raw/courses  --collection courses  --db-path data/db/courses\n"
        "  python ingest/ingest.py --source data/raw/policies --collection policies --db-path data/db/policies\n"
        "  python ingest/ingest.py --source data/raw/majors   --collection majors   --db-path data/db/majors\n",
        file=sys.stderr,
    )
    sys.exit(1)

_KS = [1, 3, 10]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rank_of_code(docs, expected_code: str) -> int | None:
    """1-indexed rank of the first doc whose metadata["code"] matches expected_code."""
    for i, doc in enumerate(docs, 1):
        if doc.metadata.get("code", "").strip() == expected_code.strip():
            return i
    return None


def _rank_of_url(docs, golden_url: str) -> int | None:
    """1-indexed rank of the first doc whose metadata["url"] matches golden_url."""
    for i, doc in enumerate(docs, 1):
        if doc.metadata.get("url", "").rstrip("/") == golden_url.rstrip("/"):
            return i
    return None


def _reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

async def run(examples: list[dict], verbose: bool = False) -> dict:
    results = []

    for ex in examples:
        question = ex.get("question", "")
        expected_code = ex.get("expected_code")
        golden_url = ex.get("golden_url")

        if not question:
            continue

        docs = await retrieve(question)

        # Determine rank by the appropriate golden signal
        rank = None
        signal_type = None
        if expected_code:
            rank = _rank_of_code(docs, expected_code)
            signal_type = "code"
        elif golden_url:
            rank = _rank_of_url(docs, golden_url)
            signal_type = "url"

        rr = _reciprocal_rank(rank)
        hits = {k: (rank is not None and rank <= k) for k in _KS}

        # Was the result retrieved by direct lookup (doc at rank 1 has the exact code)?
        direct_lookup_hit = (
            expected_code is not None
            and len(docs) > 0
            and docs[0].metadata.get("code", "").strip() == expected_code.strip()
        )

        row = {
            "id": ex.get("id", "?"),
            "question": question[:80],
            "signal_type": signal_type,
            "expected_code": expected_code,
            "golden_url": golden_url,
            "rank": rank,
            "mrr": rr,
            "direct_lookup_hit": direct_lookup_hit,
            **{f"hit@{k}": hits[k] for k in _KS},
            "n_docs_returned": len(docs),
        }
        results.append(row)

        if verbose:
            status = f"rank={rank}" if rank else "MISS"
            print(f"  [{status}] {row['question']}")

    if not results:
        return {"summary": {}, "results": []}

    # Aggregate
    n = len(results)

    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    code_results  = [r for r in results if r["signal_type"] == "code"]
    url_results   = [r for r in results if r["signal_type"] == "url"]

    summary = {
        "n": n,
        "mrr": _mean("mrr"),
        **{f"recall@{k}": round(sum(r[f"hit@{k}"] for r in results) / n, 4) for k in _KS},
        "direct_lookup_hit_rate": (
            round(sum(r["direct_lookup_hit"] for r in code_results) / len(code_results), 4)
            if code_results else None
        ),
        "by_signal": {
            "code": {
                "n": len(code_results),
                **({f"recall@{k}": round(sum(r[f"hit@{k}"] for r in code_results) / len(code_results), 4) for k in _KS} if code_results else {}),
            },
            "url": {
                "n": len(url_results),
                **({f"recall@{k}": round(sum(r[f"hit@{k}"] for r in url_results) / len(url_results), 4) for k in _KS} if url_results else {}),
            },
        },
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print("\n── Retrieval Eval Summary ───────────────────────────")
    print(f"  Examples evaluated : {summary['n']}")
    print(f"  MRR                : {summary['mrr']}")
    for k in _KS:
        print(f"  Recall@{k:<3}          : {summary[f'recall@{k}']}")
    if summary.get("direct_lookup_hit_rate") is not None:
        print(f"  Direct lookup @1   : {summary['direct_lookup_hit_rate']}")
    bs = summary.get("by_signal", {})
    for sig, stats in bs.items():
        if stats.get("n", 0) > 0:
            r1 = stats.get("recall@1", "n/a")
            print(f"    [{sig}] n={stats['n']}  recall@1={r1}")
    print("─────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_examples(paths: list[Path], limit: int | None) -> list[dict]:
    examples = []
    for p in paths:
        if not p.exists():
            print(f"WARNING: dataset not found: {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    # Only include examples that have a golden signal
    examples = [e for e in examples if e.get("expected_code") or e.get("golden_url")]
    if limit:
        examples = examples[:limit]
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality evaluation")
    parser.add_argument(
        "--dataset", nargs="+",
        default=["eval/datasets/courses.jsonl"],
        help="One or more JSONL dataset files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print per-example results")
    parser.add_argument("--out", default=None, help="Write results JSON to this path")
    args = parser.parse_args()

    examples = _load_examples([Path(p) for p in args.dataset], args.limit)
    if not examples:
        print("No examples with golden signals found. Run eval/build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating retrieval on {len(examples)} examples...")
    output = asyncio.run(run(examples, verbose=args.verbose))
    print_summary(output["summary"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
