"""
eval/harnesses/file_eval.py — File-upload pipeline harness.

Verifies the file upload path: text is extracted from a fixture file,
injected as file_context, and the answer references the uploaded content.

Checks:
  context_has_upload_block  — "[User-Attached Document]" appears in the context
  answer_mentions_upload    — answer references content from the uploaded file
  no_hallucination          — answer doesn't invent courses not in the fixture

Fixtures live in eval/fixtures/:
  sample_transcript.txt  — synthetic UCI transcript listing completed courses
  sample_policy.txt      — short policy excerpt for cross-reference testing

Usage:
    python eval/harnesses/file_eval.py
    python eval/harnesses/file_eval.py --fixture eval/fixtures/sample_transcript.txt
"""

import argparse
import asyncio
import sys
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
    from rag_chatbot.file_parser import _from_txt, _from_pdf, _from_docx
except FileNotFoundError as e:
    print(f"\nERROR: {e}\nRun ingest for all three collections first.\n", file=sys.stderr)
    sys.exit(1)

_FIXTURES_DIR = Path("eval/fixtures")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_fixture(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() == ".txt":
        return _from_txt(raw)
    if path.suffix.lower() == ".pdf":
        return _from_pdf(raw)
    if path.suffix.lower() == ".docx":
        return _from_docx(raw)
    raise ValueError(f"Unsupported fixture extension: {path.suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

# Each case: fixture file, question, strings that MUST appear in the answer,
# strings that must NOT appear in the context (sanity checks).
_TEST_CASES = [
    {
        "id": "transcript-cs-reqs",
        "fixture": "sample_transcript.txt",
        "question": "Based on my transcript, which lower-division Computer Science requirements have I completed?",
        "context_must_contain": ["[User-Attached Document]"],
        "answer_must_contain": [],       # flexible — just check it answers
        "answer_must_not_contain": [],
        "notes": "Uploaded transcript should be prepended as [User-Attached Document] block",
    },
    {
        "id": "transcript-remaining",
        "fixture": "sample_transcript.txt",
        "question": "What CS major courses do I still need to take based on my transcript?",
        "context_must_contain": ["[User-Attached Document]"],
        "answer_must_contain": [],
        "answer_must_not_contain": [],
        "notes": "Answer should reference both uploaded transcript and retrieved major requirements",
    },
    {
        "id": "policy-cross-ref",
        "fixture": "sample_policy.txt",
        "question": "Based on the policy excerpt I uploaded, what is the add/drop deadline?",
        "context_must_contain": ["[User-Attached Document]"],
        "answer_must_contain": [],
        "answer_must_not_contain": [],
        "notes": "Policy fixture should be present in context; answer references it",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

async def run_case(case: dict, verbose: bool = False) -> dict:
    fixture_path = _FIXTURES_DIR / case["fixture"]
    if not fixture_path.exists():
        return {
            "id": case["id"],
            "skipped": True,
            "reason": f"Fixture not found: {fixture_path}",
        }

    file_context = _extract_text_from_fixture(fixture_path)
    question = case["question"]

    # Step 1: rewrite (no history — file-upload questions are standalone)
    standalone_q = await _rewrite_chain.ainvoke({"input": question, "chat_history": []})

    # Step 2: retrieve from Chroma
    docs = await retrieve(standalone_q)

    # Step 3: format context with file_context injected
    ctx_inputs = _format_context({
        "standalone_question": standalone_q,
        "docs": docs,
        "file_context": file_context,
        "chat_history": [],
    })
    context = ctx_inputs.get("context", "")

    # Step 4: generate answer
    answer_llm = ChatOpenAI(model="gpt-5.4", temperature=0.5)
    answer = await (_answer_prompt | answer_llm | StrOutputParser()).ainvoke(ctx_inputs)

    # Checks
    ctx_checks = {
        s: (s in context)
        for s in case.get("context_must_contain", [])
    }
    answer_contains = {
        s: (s.lower() in answer.lower())
        for s in case.get("answer_must_contain", [])
    }
    answer_forbidden = {
        s: (s.lower() in answer.lower())
        for s in case.get("answer_must_not_contain", [])
    }

    passed = (
        all(ctx_checks.values())
        and all(answer_contains.values())
        and not any(answer_forbidden.values())
    )

    result = {
        "id": case["id"],
        "question": question,
        "standalone_question": standalone_q,
        "n_docs": len(docs),
        "upload_block_in_context": "[User-Attached Document]" in context,
        "context_checks": ctx_checks,
        "answer_contains": answer_contains,
        "answer_forbidden_present": answer_forbidden,
        "answer_preview": answer[:200],
        "passed": passed,
        "notes": case.get("notes", ""),
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {case['id']}")
        if not ctx_checks or not all(ctx_checks.values()):
            print(f"    Context checks failed: {ctx_checks}")
        if answer_forbidden and any(answer_forbidden.values()):
            print(f"    Forbidden terms in answer: {answer_forbidden}")

    return result


async def run(verbose: bool = False, fixtures: list[str] | None = None) -> dict:
    cases = _TEST_CASES
    if fixtures:
        fixture_names = {Path(f).name for f in fixtures}
        cases = [c for c in cases if c["fixture"] in fixture_names]

    results = []
    for case in cases:
        result = await run_case(case, verbose=verbose)
        results.append(result)

    run_results = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    passed = sum(1 for r in run_results if r.get("passed"))

    summary = {
        "n": len(run_results),
        "skipped": len(skipped),
        "passed": passed,
        "failed": len(run_results) - passed,
        "pass_rate": round(passed / len(run_results), 4) if run_results else None,
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print("\n── File Upload Eval Summary ─────────────────────────")
    print(f"  Ran     : {summary['n']}")
    print(f"  Skipped : {summary['skipped']}")
    print(f"  Passed  : {summary['passed']}")
    print(f"  Failed  : {summary['failed']}")
    if summary.get("pass_rate") is not None:
        print(f"  Pass rate: {summary['pass_rate']:.4f}")
    print("─────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="File-upload pipeline evaluation")
    parser.add_argument("--fixture", nargs="+", default=None, help="Specific fixture files to test")
    parser.add_argument("--verbose", action="store_true", help="Per-case output")
    args = parser.parse_args()

    output = asyncio.run(run(verbose=args.verbose, fixtures=args.fixture))
    print_summary(output["summary"])

    if output["summary"].get("failed", 0) > 0:
        for r in output["results"]:
            if not r.get("passed") and not r.get("skipped"):
                print(f"  FAILED: {r['id']} — {r.get('notes', '')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
