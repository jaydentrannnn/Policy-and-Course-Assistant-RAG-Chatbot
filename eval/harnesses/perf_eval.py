"""
eval/harnesses/perf_eval.py — Latency and cost benchmarking.

Measures end-to-end latency (wall clock) and token usage for a sample of
questions. OpenAI response metadata provides token counts; cost is estimated
using current pricing.

Metrics per example:
  latency_s         — wall-clock seconds from question to final token
  prompt_tokens     — tokens in the full prompt sent to the final LLM
  completion_tokens — tokens in the answer
  total_tokens      — prompt + completion
  est_cost_usd      — estimated cost (see _PRICING below)

Summary: p50, p95 latency; mean token counts; total estimated cost.

Usage:
    python eval/harnesses/perf_eval.py --limit 20
    python eval/harnesses/perf_eval.py --dataset eval/datasets/hard_cases.jsonl --limit 10
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

try:
    from rag_chatbot.retriever import retrieve
    from rag_chatbot.chain import _rewrite_chain, _answer_prompt, _format_context
except FileNotFoundError as e:
    print(f"\nERROR: {e}\nRun ingest for all three collections first.\n", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Pricing (USD per 1k tokens) — update as model pricing changes
# ──────────────────────────────────────────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {
        "input":  0.002,   # $ per 1k input tokens
        "output": 0.008,   # $ per 1k output tokens
    },
    "gpt-5.4-mini": {
        "input":  0.00015,
        "output": 0.0006,
    },
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _PRICING.get(model, _PRICING["gpt-5.4"])
    return (
        prompt_tokens * pricing["input"] / 1000
        + completion_tokens * pricing["output"] / 1000
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

async def run(examples: list[dict], verbose: bool = False) -> dict:
    answer_llm = ChatOpenAI(model="gpt-5.4", temperature=0.5)
    results = []

    for ex in examples:
        question = ex.get("question", "")
        if not question or ex.get("type") == "multiturn":
            continue

        t_start = time.perf_counter()

        try:
            # Rewrite
            standalone_q = await _rewrite_chain.ainvoke(
                {"input": question, "chat_history": []}
            )

            # Retrieve
            docs = await retrieve(standalone_q)

            # Format context
            ctx_inputs = _format_context({
                "standalone_question": standalone_q,
                "docs": docs,
                "file_context": None,
                "chat_history": [],
            })

            # Generate — use .generate() to access token usage metadata
            messages = _answer_prompt.format_messages(**ctx_inputs)
            gen_result = await answer_llm.agenerate([messages])

        except Exception as exc:
            results.append({"id": ex.get("id", "?"), "question": question[:80], "error": str(exc)})
            continue

        latency = round(time.perf_counter() - t_start, 3)

        # Extract token usage from LangChain generation result
        llm_output = gen_result.llm_output or {}
        usage = llm_output.get("token_usage", {})
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
        est_cost          = _estimate_cost("gpt-5.4", prompt_tokens, completion_tokens)

        row = {
            "id": ex.get("id", "?"),
            "question": question[:80],
            "latency_s": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "est_cost_usd": round(est_cost, 6),
            "n_docs": len(docs),
        }
        results.append(row)

        if verbose:
            print(f"  {row['question'][:60]}")
            print(f"    latency={latency}s  tokens={total_tokens}  cost=${est_cost:.5f}")

    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"summary": {}, "results": results}

    latencies = sorted(r["latency_s"] for r in ok)
    n = len(latencies)

    def _pct(vals, p):
        idx = min(int(len(vals) * p / 100), len(vals) - 1)
        return vals[idx]

    total_cost = sum(r["est_cost_usd"] for r in ok)

    summary = {
        "n": n,
        "latency_p50": _pct(latencies, 50),
        "latency_p95": _pct(latencies, 95),
        "latency_mean": round(sum(latencies) / n, 3),
        "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in ok) / n),
        "completion_tokens_mean": round(sum(r["completion_tokens"] for r in ok) / n),
        "total_tokens_mean": round(sum(r["total_tokens"] for r in ok) / n),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_query_usd": round(total_cost / n, 6),
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print("\n── Performance Eval Summary ─────────────────────────")
    print(f"  Examples      : {summary['n']}")
    print(f"  Latency p50   : {summary['latency_p50']}s")
    print(f"  Latency p95   : {summary['latency_p95']}s")
    print(f"  Latency mean  : {summary['latency_mean']}s")
    print(f"  Tokens mean   : {summary['total_tokens_mean']} (prompt={summary['prompt_tokens_mean']}, completion={summary['completion_tokens_mean']})")
    print(f"  Cost/query    : ${summary['cost_per_query_usd']:.5f}")
    print(f"  Total cost    : ${summary['total_cost_usd']:.5f}")
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
    if limit:
        examples = examples[:limit]
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency and cost benchmarking")
    parser.add_argument(
        "--dataset", nargs="+",
        default=["eval/datasets/courses.jsonl", "eval/datasets/hard_cases.jsonl"],
        help="One or more JSONL dataset files",
    )
    parser.add_argument("--limit",   type=int, default=20,   help="Max examples (default: 20)")
    parser.add_argument("--verbose", action="store_true",    help="Per-example output")
    parser.add_argument("--out",     default=None,           help="Write results CSV to this path")
    args = parser.parse_args()

    examples = _load_examples([Path(p) for p in args.dataset], args.limit)
    if not examples:
        print("No examples found. Run eval/build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking {len(examples)} examples...")
    output = asyncio.run(run(examples, verbose=args.verbose))
    print_summary(output["summary"])

    out_path = Path(args.out) if args.out else Path(f"eval/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/perf.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = [r for r in output["results"] if "error" not in r]
    if ok:
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ok[0].keys())
            writer.writeheader()
            writer.writerows(ok)
        print(f"Per-query CSV written to {out_path}")


if __name__ == "__main__":
    main()
