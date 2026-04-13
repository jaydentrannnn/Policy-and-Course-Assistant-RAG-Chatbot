"""
eval/harnesses/e2e_eval.py — End-to-end generation quality harness.

Calls chain components explicitly (not _base_chain) so intermediates are
accessible without modifying production code:
  1. _rewrite_chain  → standalone_question
  2. retrieve()      → docs
  3. _format_context → context string
  4. _answer_prompt + LLM → answer

Metrics:
  citation_ok      — all URLs in answer are in the retrieved docs set
  field_match      — regex match of expected_value in answer (deterministic)
  faithfulness     — LLM judge 1–5: does answer stay within retrieved context?
  relevance        — LLM judge 1–5: does answer address the question?

Results are persisted to --out (default: eval/runs/<timestamp>/e2e_results.jsonl).

Usage:
    python eval/harnesses/e2e_eval.py --dataset eval/datasets/courses.jsonl --limit 20
    python eval/harnesses/e2e_eval.py --dataset eval/datasets/hard_cases.jsonl --judge
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

try:
    from rag_chatbot.retriever import retrieve
    from rag_chatbot.chain import _rewrite_chain, _answer_prompt, _format_context
except FileNotFoundError as e:
    print(
        f"\nERROR: {e}\n"
        "Chroma DBs must be built before running e2e eval.\n"
        "Run: python ingest/ingest.py for all three collections.\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# LLM judge
# ──────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG chatbot answer for faithfulness.

Question: {question}

Retrieved context:
{context}

Answer:
{answer}

Score the answer from 1 to 5:
5 = Every factual claim is directly supported by the context.
4 = Almost all claims supported; minor extrapolation.
3 = Mostly supported but one unsupported claim.
2 = Several unsupported or contradicted claims.
1 = Answer is fabricated or contradicts the context.

Reply with only a single integer (1–5).
"""

_RELEVANCE_PROMPT = """\
You are evaluating a RAG chatbot answer for relevance.

Question: {question}
Answer: {answer}

Score the answer from 1 to 5:
5 = Directly and completely addresses the question.
4 = Addresses the question with minor gaps.
3 = Partially addresses the question.
2 = Tangentially related but mostly misses the point.
1 = Irrelevant or refuses to answer without justification.

Reply with only a single integer (1–5).
"""


async def _judge_score(prompt: str, llm: ChatOpenAI) -> int | None:
    try:
        resp = await (llm | StrOutputParser()).ainvoke([HumanMessage(content=prompt)])
        m = re.search(r"[1-5]", resp.strip())
        return int(m.group()) if m else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Citation check
# ──────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\)\]\"']+")


def _check_citations(answer: str, docs) -> bool:
    """All URLs mentioned in the answer must come from retrieved doc metadata."""
    urls_in_answer = set(_URL_RE.findall(answer))
    if not urls_in_answer:
        return True  # no citations = vacuously ok
    doc_urls = {d.metadata.get("url", "").rstrip("/") for d in docs}
    return all(u.rstrip("/") in doc_urls for u in urls_in_answer)


# ──────────────────────────────────────────────────────────────────────────────
# Field match
# ──────────────────────────────────────────────────────────────────────────────

def _check_field_match(answer: str, expected_value: str) -> bool:
    """Check if the expected value appears verbatim (case-insensitive) in the answer."""
    return expected_value.lower() in answer.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

async def run(
    examples: list[dict],
    use_judge: bool = False,
    judge_model: str = "gpt-5.4-mini",
    verbose: bool = False,
) -> dict:
    judge_llm = ChatOpenAI(model=judge_model, temperature=0) if use_judge else None
    answer_llm = ChatOpenAI(model="gpt-5.4", temperature=0.5)
    results = []

    for ex in examples:
        question = ex.get("question", "")
        if not question:
            continue

        try:
            # Step 1: rewrite
            standalone_q = await _rewrite_chain.ainvoke(
                {"input": question, "chat_history": []}
            )

            # Step 2: retrieve
            docs = await retrieve(standalone_q)

            # Step 3: format context
            file_context = ex.get("file_context")
            ctx_inputs = _format_context({
                "standalone_question": standalone_q,
                "docs": docs,
                "file_context": file_context,
                "chat_history": [],
            })

            # Step 4: generate answer
            answer = await (
                _answer_prompt | answer_llm | StrOutputParser()
            ).ainvoke(ctx_inputs)

        except Exception as exc:
            results.append({
                "id": ex.get("id", "?"),
                "question": question[:80],
                "error": str(exc),
            })
            if verbose:
                print(f"  [ERROR] {question[:60]}: {exc}")
            continue

        # Deterministic metrics
        citation_ok = _check_citations(answer, docs)
        expected_value = ex.get("expected_value")
        field_match = _check_field_match(answer, expected_value) if expected_value else None

        # LLM judge metrics
        faithfulness = None
        relevance = None
        if use_judge and judge_llm:
            context_text = ctx_inputs.get("context", "")[:3000]  # truncate for judge
            faithfulness = await _judge_score(
                _FAITHFULNESS_PROMPT.format(
                    question=question, context=context_text, answer=answer
                ),
                judge_llm,
            )
            relevance = await _judge_score(
                _RELEVANCE_PROMPT.format(question=question, answer=answer),
                judge_llm,
            )

        row = {
            "id": ex.get("id", "?"),
            "question": question[:80],
            "standalone_question": standalone_q[:100],
            "n_docs": len(docs),
            "answer_preview": answer[:200],
            "citation_ok": citation_ok,
            "field_match": field_match,
            "faithfulness": faithfulness,
            "relevance": relevance,
        }
        results.append(row)

        if verbose:
            flags = []
            if not citation_ok:
                flags.append("BAD_CITATION")
            if field_match is False:
                flags.append("FIELD_MISS")
            status = " ".join(flags) if flags else "OK"
            print(f"  [{status}] {row['question']}")
            if faithfulness:
                print(f"       faith={faithfulness} rel={relevance}")

    if not results:
        return {"summary": {}, "results": []}

    ok_results = [r for r in results if "error" not in r]
    n = len(ok_results)

    def _pct(key, filter_none=True):
        vals = [r[key] for r in ok_results if r.get(key) is not None] if filter_none else [r[key] for r in ok_results]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None

    def _mean(key):
        vals = [r[key] for r in ok_results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n": n,
        "errors": len(results) - n,
        "citation_ok_rate": _pct("citation_ok"),
        "field_match_rate": _pct("field_match"),
        "faithfulness_mean": _mean("faithfulness"),
        "relevance_mean": _mean("relevance"),
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print("\n── E2E Eval Summary ─────────────────────────────────")
    print(f"  Examples evaluated : {summary['n']} ({summary.get('errors', 0)} errors)")
    if summary.get("citation_ok_rate") is not None:
        print(f"  Citation accuracy  : {summary['citation_ok_rate']:.4f}")
    if summary.get("field_match_rate") is not None:
        print(f"  Field match rate   : {summary['field_match_rate']:.4f}")
    if summary.get("faithfulness_mean") is not None:
        print(f"  Faithfulness (1-5) : {summary['faithfulness_mean']:.2f}")
    if summary.get("relevance_mean") is not None:
        print(f"  Relevance (1-5)    : {summary['relevance_mean']:.2f}")
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
                    ex = json.loads(line)
                    if ex.get("type") != "multiturn":  # skip multi-turn
                        examples.append(ex)
    if limit:
        examples = examples[:limit]
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end generation quality evaluation")
    parser.add_argument(
        "--dataset", nargs="+",
        default=["eval/datasets/courses.jsonl"],
        help="One or more JSONL dataset files",
    )
    parser.add_argument("--limit",       type=int,   default=None,          help="Max examples")
    parser.add_argument("--judge",       action="store_true",                help="Enable LLM judge scoring")
    parser.add_argument("--judge-model", default="gpt-5.4-mini",            help="Model for judge")
    parser.add_argument("--verbose",     action="store_true",                help="Per-example output")
    parser.add_argument(
        "--out", default=None,
        help="Write results JSONL to this path (default: eval/runs/<ts>/e2e_results.jsonl)"
    )
    args = parser.parse_args()

    examples = _load_examples([Path(p) for p in args.dataset], args.limit)
    if not examples:
        print("No examples found. Run eval/build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(examples)} examples (judge={'on' if args.judge else 'off'})...")
    output = asyncio.run(run(examples, use_judge=args.judge, judge_model=args.judge_model, verbose=args.verbose))
    print_summary(output["summary"])

    out_path = Path(args.out) if args.out else Path(f"eval/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/e2e_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in output["results"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_path.parent / "e2e_summary.json"
    summary_path.write_text(json.dumps(output["summary"], indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
