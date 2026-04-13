"""
eval/harnesses/multiturn_eval.py — Conversational correctness harness.

Tests rewrite-chain behaviour over scripted multi-turn conversations from
eval/datasets/hard_cases.jsonl (type == "multiturn").

Each example has:
  turns                  list[str]   — the user messages in order
  rewrite_must_contain   list[str]   — strings that MUST appear in the rewritten turn-2 question
  rewrite_must_not_contain list[str] — strings that must NOT appear (polarity inversion guard)
  notes                  str         — human explanation of what's being tested

Run as a pytest module for pass/fail reporting:
    pytest eval/harnesses/multiturn_eval.py -v

Or run standalone for a prose summary:
    python eval/harnesses/multiturn_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when run as a standalone script or via pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from dotenv import load_dotenv

load_dotenv()

# chain.py imports retriever.py, which loads all three Chroma collections at module
# level. If the DBs don't exist yet, wrap this so pytest skips cleanly instead of
# crashing the entire collection phase.
_CHAIN_AVAILABLE = False
_CHAIN_ERROR = ""
try:
    from rag_chatbot.chain import _rewrite_chain
    _CHAIN_AVAILABLE = True
except FileNotFoundError as _e:
    _CHAIN_ERROR = str(_e)

pytestmark = pytest.mark.skipif(
    not _CHAIN_AVAILABLE,
    reason=(
        f"Chroma DBs not built — run ingest for all three collections first. ({_CHAIN_ERROR})"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Load hard cases
# ──────────────────────────────────────────────────────────────────────────────

_HARD_CASES_PATH = Path("eval/datasets/hard_cases.jsonl")


def _load_multiturn_cases() -> list[dict]:
    if not _HARD_CASES_PATH.exists():
        return []
    cases = []
    with open(_HARD_CASES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("type") == "multiturn":
                cases.append(ex)
    return cases


_CASES = _load_multiturn_cases()


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

async def _run_conversation(turns: list[str]) -> list[str]:
    """Run a scripted conversation and return the rewritten form of each user turn."""
    if not _CHAIN_AVAILABLE:
        raise RuntimeError(f"Chroma DBs not available: {_CHAIN_ERROR}")
    from langchain_core.messages import AIMessage, HumanMessage

    history = []
    rewrites = []
    for turn in turns:
        rewritten = await _rewrite_chain.ainvoke({"input": turn, "chat_history": history})
        rewrites.append(rewritten)
        # Add this turn + a placeholder assistant reply so history grows naturally
        history.append(HumanMessage(content=turn))
        history.append(AIMessage(content="[placeholder]"))
    return rewrites


def _check_case(case: dict) -> dict:
    turns = case["turns"]
    must_contain     = [s.lower() for s in case.get("rewrite_must_contain", [])]
    must_not_contain = [s.lower() for s in case.get("rewrite_must_not_contain", [])]

    rewrites = asyncio.run(_run_conversation(turns))
    # The rewrite under test is always the LAST turn
    final_rewrite = rewrites[-1].lower()

    missing  = [s for s in must_contain     if s not in final_rewrite]
    present  = [s for s in must_not_contain if s in final_rewrite]

    return {
        "id":              case.get("id", "?"),
        "notes":           case.get("notes", ""),
        "turns":           turns,
        "final_rewrite":   rewrites[-1],
        "missing":         missing,
        "present":         present,
        "passed":          not missing and not present,
    }


# ──────────────────────────────────────────────────────────────────────────────
# pytest parametrisation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", _CASES, ids=[c.get("id", f"case-{i}") for i, c in enumerate(_CASES)])
def test_multiturn_rewrite(case: dict) -> None:
    """Rewrite chain must preserve polarity and resolve pronouns."""
    result = _check_case(case)

    assert not result["missing"], (
        f"[{case['id']}] Rewritten question is missing required terms: {result['missing']}\n"
        f"  Final rewrite: {result['final_rewrite']!r}\n"
        f"  Notes: {case.get('notes', '')}"
    )
    assert not result["present"], (
        f"[{case['id']}] Rewritten question contains forbidden terms (polarity inversion?): {result['present']}\n"
        f"  Final rewrite: {result['final_rewrite']!r}\n"
        f"  Notes: {case.get('notes', '')}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cases = _load_multiturn_cases()
    if not cases:
        print("No multi-turn cases found in eval/datasets/hard_cases.jsonl", file=sys.stderr)
        sys.exit(1)

    passed = 0
    failed = 0
    for case in cases:
        result = _check_case(case)
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {case.get('id', '?')} — {case.get('notes', '')}")
        if not result["passed"]:
            print(f"    Rewrite : {result['final_rewrite']!r}")
            if result["missing"]:
                print(f"    Missing : {result['missing']}")
            if result["present"]:
                print(f"    Forbidden present: {result['present']}")

    print(f"\n{passed}/{passed + failed} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
