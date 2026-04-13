"""
eval/harnesses/router_eval.py — Router accuracy harness.

Tests the _RouterDecision LLM call in isolation. Does NOT require Chroma DBs
(avoids importing rag_chatbot.retriever, which loads all three collections at
import time). The router logic is reproduced here from retriever.py.

Metrics:
  collection_f1     — macro-averaged F1 over the three collection labels
  requires_acc      — accuracy of requires_full_requirements prediction
  major_kw_match    — fraction of examples where predicted major_keyword
                      fuzzy-matches the expected one (case-insensitive contains)

Usage:
    python eval/harnesses/router_eval.py --dataset eval/datasets/courses.jsonl
    python eval/harnesses/router_eval.py --dataset eval/datasets/hard_cases.jsonl --limit 20
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Inline router (mirrors retriever.py exactly — no Chroma dependency)
# ──────────────────────────────────────────────────────────────────────────────

class _RouterDecision(BaseModel):
    collections: list[Literal["courses", "policies", "majors"]]
    major_keyword: str | None = None
    requires_full_requirements: bool = False


_router_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0).with_structured_output(
    _RouterDecision
)

_ROUTER_SYSTEM = """\
You are a query router for a UCI student academic assistant.
Decide which knowledge bases to search based on the student's question.

Collections:
- courses   → specific course info, prerequisites, corequisites, units, descriptions, restrictions
- policies  → academic rules, grading, add/drop deadlines, academic integrity, probation, transfer credit
- majors    → major/minor requirements, degree plans, required courses for a specific major or minor

Rules:
- Return only the collections that are clearly relevant.
- When a question asks about required courses for a major/minor, return BOTH "majors" AND "courses" —
  the major page lists required course codes, and the courses collection has their full details.
- When a question spans multiple topics (e.g. "does course X count for my major"), return multiple.
- If genuinely unsure, return all three — over-retrieval is better than missing context.

major_keyword: If the question is about a specific major or minor, set this to the program name
exactly as it would appear in the UCI catalogue (e.g. "Computer Science", "Informatics",
"Mathematics", "Electrical Engineering"). Leave null if no specific program is mentioned.
This is used to filter the majors collection to only relevant programme pages.

requires_full_requirements: Set to true when the student is asking what courses are
required for a specific major or minor, OR when they are asking whether a specific course
is required/mandatory in a major or minor. Examples that should be true:
  "what courses do I need for the CS major?"
  "list the lower-division requirements for Computer Science"
  "what upper-division courses are required for Math?"
  "do I have to take COMPSCI 161 as a CS major?"
  "is COMPSCI 161 required for the Computer Science major?"
  "does the CS major require COMPSCI 163?"
  "is ICS 6B mandatory for the Informatics major?"
Set to false for questions about major overview, specialisations, admissions, sample
plans, career paths, or anything that is not about whether specific courses are required.
"""


async def _route(question: str) -> tuple[list[str], str | None, bool]:
    decision: _RouterDecision = await _router_llm.ainvoke(
        [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
    )
    return decision.collections, decision.major_keyword, decision.requires_full_requirements


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

ALL_COLLECTIONS = {"courses", "policies", "majors"}


def _collection_f1(predicted: list[str], expected: list[str]) -> float:
    p_set = set(predicted) & ALL_COLLECTIONS
    e_set = set(expected) & ALL_COLLECTIONS
    if not e_set and not p_set:
        return 1.0
    if not e_set or not p_set:
        return 0.0
    tp = len(p_set & e_set)
    precision = tp / len(p_set)
    recall = tp / len(e_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _major_kw_match(predicted: str | None, expected: str | None) -> bool | None:
    """None means the example has no expected major_keyword (not applicable)."""
    if expected is None:
        return None
    if predicted is None:
        return False
    return expected.lower() in predicted.lower() or predicted.lower() in expected.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

async def run(examples: list[dict], verbose: bool = False) -> dict:
    results = []

    for ex in examples:
        question = ex.get("question") or ex.get("turns", [""])[0]
        if not question:
            continue

        pred_collections, pred_kw, pred_req = await _route(question)

        expected_collections = ex.get("collections", [])
        expected_kw = ex.get("major_keyword")
        expected_req = ex.get("requires_full_requirements")

        f1 = _collection_f1(pred_collections, expected_collections)
        kw_match = _major_kw_match(pred_kw, expected_kw)
        req_correct = (pred_req == expected_req) if expected_req is not None else None

        row = {
            "id": ex.get("id", "?"),
            "question": question[:80],
            "pred_collections": sorted(pred_collections),
            "exp_collections": sorted(expected_collections),
            "collection_f1": round(f1, 3),
            "pred_major_kw": pred_kw,
            "exp_major_kw": expected_kw,
            "major_kw_match": kw_match,
            "pred_req_full": pred_req,
            "exp_req_full": expected_req,
            "req_full_correct": req_correct,
        }
        results.append(row)

        if verbose:
            status = "OK" if f1 == 1.0 else "FAIL"
            print(f"  [{status}] {row['question']}")
            if f1 < 1.0:
                print(f"       collections: pred={pred_collections} exp={expected_collections}")
            if req_correct is False:
                print(f"       req_full:    pred={pred_req} exp={expected_req}")

    # Summary
    f1_scores = [r["collection_f1"] for r in results]
    req_results = [r["req_full_correct"] for r in results if r["req_full_correct"] is not None]
    kw_results  = [r["major_kw_match"]   for r in results if r["major_kw_match"]   is not None]

    summary = {
        "n": len(results),
        "collection_f1_mean": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "collection_exact_match": round(sum(1 for s in f1_scores if s == 1.0) / len(f1_scores), 4) if f1_scores else 0.0,
        "requires_full_req_accuracy": round(sum(req_results) / len(req_results), 4) if req_results else None,
        "major_kw_match_rate": round(sum(kw_results) / len(kw_results), 4) if kw_results else None,
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print("\n── Router Eval Summary ──────────────────────────────")
    print(f"  Examples evaluated :  {summary['n']}")
    print(f"  Collection F1 (mean): {summary['collection_f1_mean']:.4f}")
    print(f"  Collection exact    : {summary['collection_exact_match']:.4f}")
    if summary["requires_full_req_accuracy"] is not None:
        print(f"  Requires-full-req   : {summary['requires_full_req_accuracy']:.4f}")
    if summary["major_kw_match_rate"] is not None:
        print(f"  Major keyword match : {summary['major_kw_match_rate']:.4f}")
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
    parser = argparse.ArgumentParser(description="Router accuracy evaluation")
    parser.add_argument(
        "--dataset", nargs="+",
        default=["eval/datasets/courses.jsonl", "eval/datasets/majors.jsonl",
                 "eval/datasets/policies.jsonl", "eval/datasets/hard_cases.jsonl"],
        help="One or more JSONL dataset files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print per-example results")
    parser.add_argument("--out", default=None, help="Write results JSON to this path")
    args = parser.parse_args()

    examples = _load_examples([Path(p) for p in args.dataset], args.limit)
    if not examples:
        print("No examples found. Run eval/build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(examples)} examples...")
    output = asyncio.run(run(examples, verbose=args.verbose))
    print_summary(output["summary"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
