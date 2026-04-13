"""
eval/build_dataset.py — Auto-generate evaluation datasets from data/raw/.

Walks data/raw/{courses,majors,policies}/ and emits three JSONL files:
  eval/datasets/courses.jsonl   — prerequisite / units / repeatability questions
  eval/datasets/majors.jsonl    — major/minor requirement questions
  eval/datasets/policies.jsonl  — policy-section questions

Run with the venv active after crawling:
    python eval/build_dataset.py

Options:
  --raw-dir      path to data/raw/ (default: data/raw)
  --out-dir      path to eval/datasets/ (default: eval/datasets)
  --max-courses  max course examples to emit (default: 300)
  --max-majors   max major examples to emit (default: 150)
  --max-policies max policy examples to emit (default: 120)
  --seed         random seed for sampling (default: 42)
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Course code regex (same pattern as retriever.py)
# ──────────────────────────────────────────────────────────────────────────────

_COURSE_CODE_RE = re.compile(
    r"\b((?:[A-Z]&[A-Z]\s[A-Z]{2,8}|[A-Z]{2,8}(?:\s[A-Z]{2,8})?(?:/[A-Z]{2,8})?)\s+[A-Z]?\d+[A-Z]{0,2})\b"
)


def _extract_course_codes(text: str) -> list[str]:
    return list(dict.fromkeys(
        re.sub(r"\s+", " ", m).strip()
        for m in _COURSE_CODE_RE.findall(text)
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Courses dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_courses(raw_courses: Path, max_examples: int, rng: random.Random) -> list[dict]:
    examples: list[dict] = []

    for json_file in sorted(raw_courses.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("type") != "course_page":
            continue

        page_url = data.get("url", "")
        for course in data.get("courses", []):
            code = (course.get("code") or "").strip()
            if not code:
                continue

            course_url = course.get("url") or page_url

            # Prerequisite question
            prereq = (course.get("prerequisite") or "").strip()
            if prereq:
                safe_id = re.sub(r"[^A-Za-z0-9]", "-", code)
                examples.append({
                    "id": f"prereq-{safe_id}",
                    "question": f"What are the prerequisites for {code}?",
                    "type": "course_prereq",
                    "expected_code": code,
                    "expected_field": "prerequisite",
                    "expected_value": prereq,
                    "collections": ["courses"],
                    "requires_full_requirements": False,
                    "golden_url": course_url,
                })

            # Units question
            units = (course.get("units") or "").strip()
            if units:
                safe_id = re.sub(r"[^A-Za-z0-9]", "-", code)
                examples.append({
                    "id": f"units-{safe_id}",
                    "question": f"How many units is {code}?",
                    "type": "course_units",
                    "expected_code": code,
                    "expected_field": "units",
                    "expected_value": units,
                    "collections": ["courses"],
                    "requires_full_requirements": False,
                    "golden_url": course_url,
                })

            # Repeatability question
            repeatability = (course.get("repeatability") or "").strip()
            if repeatability:
                safe_id = re.sub(r"[^A-Za-z0-9]", "-", code)
                examples.append({
                    "id": f"repeat-{safe_id}",
                    "question": f"Can {code} be repeated for credit?",
                    "type": "course_repeatability",
                    "expected_code": code,
                    "expected_field": "repeatability",
                    "expected_value": repeatability,
                    "collections": ["courses"],
                    "requires_full_requirements": False,
                    "golden_url": course_url,
                })

    rng.shuffle(examples)
    return examples[:max_examples]


# ──────────────────────────────────────────────────────────────────────────────
# Majors dataset
# ──────────────────────────────────────────────────────────────────────────────

def _parse_program_name(title: str) -> str | None:
    """Extract a clean program name from a policy_page title.

    e.g. "Computer Science, B.S." → "Computer Science"
         "Informatics Minor"       → "Informatics"
    """
    # Strip degree designations and clean up
    name = re.sub(
        r",?\s*(B\.S\.|B\.A\.|B\.F\.A\.|M\.S\.|Ph\.D\.|Minor|Major|Program|Department)\.?",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    # Must be non-trivial
    return name if len(name) >= 3 else None


def build_majors(raw_majors: Path, max_examples: int, rng: random.Random) -> list[dict]:
    examples: list[dict] = []

    for json_file in sorted(raw_majors.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("type") != "policy_page":
            continue

        page_url = data.get("url", "")
        page_title = (data.get("title") or "").strip()
        sections = data.get("sections", [])

        # Find sections that mention requirements
        req_sections = [
            s for s in sections
            if re.search(r"(major|minor)\s+requirement", s.get("heading", ""), re.IGNORECASE)
        ]
        if not req_sections:
            continue

        # Try to infer the program name from title or first heading
        program_name = _parse_program_name(page_title)
        if not program_name and sections:
            program_name = _parse_program_name(sections[0].get("heading", ""))
        if not program_name:
            continue

        # Determine if it's a major or minor
        is_minor = bool(re.search(r"minor", page_title, re.IGNORECASE))
        degree_type = "minor" if is_minor else "major"

        # Question 1: full requirements list
        examples.append({
            "id": f"req-{hashlib.sha1(page_url.encode()).hexdigest()[:10]}",
            "question": f"What courses are required for the {program_name} {degree_type}?",
            "type": "major_requirements",
            "major_keyword": program_name,
            "requires_full_requirements": True,
            "collections": ["majors", "courses"],
            "golden_url": page_url,
        })

        # Question 2: "is course X required?" for a random required course
        all_content = " ".join(s.get("content", "") for s in req_sections)
        codes_in_requirements = _extract_course_codes(all_content)
        if codes_in_requirements:
            sampled_code = rng.choice(codes_in_requirements)
            examples.append({
                "id": f"isreq-{hashlib.sha1((page_url + sampled_code).encode()).hexdigest()[:10]}",
                "question": f"Is {sampled_code} required for the {program_name} {degree_type}?",
                "type": "course_required_for_major",
                "major_keyword": program_name,
                "requires_full_requirements": True,
                "expected_code": sampled_code,
                "collections": ["majors", "courses"],
                "golden_url": page_url,
            })

    rng.shuffle(examples)
    return examples[:max_examples]


# ──────────────────────────────────────────────────────────────────────────────
# Policies dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_policies(raw_policies: Path, max_examples: int, rng: random.Random) -> list[dict]:
    examples: list[dict] = []

    # Headings to skip — too generic or navigation-only
    _SKIP_HEADINGS = {
        "", "contents", "menu", "navigation", "home", "back", "skip to main content",
        "search", "contact", "site map", "accessibility", "privacy", "copyright",
    }

    for json_file in sorted(raw_policies.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("type") != "policy_page":
            continue

        page_url = data.get("url", "")
        for section in data.get("sections", []):
            heading = (section.get("heading") or "").strip()
            content = (section.get("content") or "").strip()

            if not heading or not content:
                continue
            if heading.lower() in _SKIP_HEADINGS:
                continue
            if len(content) < 50:  # skip trivially short sections
                continue

            ex_id = hashlib.sha1(f"{page_url}::{heading}".encode()).hexdigest()[:12]
            examples.append({
                "id": f"policy-{ex_id}",
                "question": f"What is UCI's policy on {heading}?",
                "type": "policy_section",
                "collections": ["policies"],
                "requires_full_requirements": False,
                "golden_url": page_url,
                "golden_heading": heading,
            })

    rng.shuffle(examples)
    return examples[:max_examples]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval datasets from data/raw/")
    parser.add_argument("--raw-dir",      default="data/raw",      help="Path to data/raw/")
    parser.add_argument("--out-dir",      default="eval/datasets",  help="Output directory")
    parser.add_argument("--max-courses",  type=int, default=300,    help="Max course examples")
    parser.add_argument("--max-majors",   type=int, default=150,    help="Max major examples")
    parser.add_argument("--max-policies", type=int, default=120,    help="Max policy examples")
    parser.add_argument("--seed",         type=int, default=42,     help="Random seed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    if not raw_dir.exists():
        print(
            f"ERROR: Raw data directory '{raw_dir}' does not exist.\n"
            "Run the crawler first:\n"
            "  python crawler/crawler.py https://catalogue.uci.edu/allcourses --type course_catalog\n"
            "  python crawler/crawler.py https://catalogue.uci.edu/informationforadmittedstudents "
            "--type policy --output data/raw/policies\n"
            "  (and so on — see CLAUDE.md for the full list)",
            file=sys.stderr,
        )
        sys.exit(1)

    results = {}

    # Courses
    courses_dir = raw_dir / "courses"
    if courses_dir.exists():
        examples = build_courses(courses_dir, args.max_courses, rng)
        _write_jsonl(out_dir / "courses.jsonl", examples)
        results["courses"] = len(examples)
        print(f"  courses.jsonl     {len(examples):>4} examples")
    else:
        print(f"  courses.jsonl     SKIPPED (no {courses_dir})")
        results["courses"] = 0

    # Majors
    majors_dir = raw_dir / "majors"
    if majors_dir.exists():
        examples = build_majors(majors_dir, args.max_majors, rng)
        _write_jsonl(out_dir / "majors.jsonl", examples)
        results["majors"] = len(examples)
        print(f"  majors.jsonl      {len(examples):>4} examples")
    else:
        print(f"  majors.jsonl      SKIPPED (no {majors_dir})")
        results["majors"] = 0

    # Policies
    policies_dir = raw_dir / "policies"
    if policies_dir.exists():
        examples = build_policies(policies_dir, args.max_policies, rng)
        _write_jsonl(out_dir / "policies.jsonl", examples)
        results["policies"] = len(examples)
        print(f"  policies.jsonl    {len(examples):>4} examples")
    else:
        print(f"  policies.jsonl    SKIPPED (no {policies_dir})")
        results["policies"] = 0

    total = sum(results.values())
    print(f"\nDone. {total} auto-generated examples written to {out_dir}/")
    hard_path = out_dir / "hard_cases.jsonl"
    if hard_path.exists():
        hard_count = sum(1 for _ in hard_path.open(encoding="utf-8"))
        print(f"  hard_cases.jsonl  {hard_count:>4} examples (hand-curated, already present)")
    else:
        print("  hard_cases.jsonl  (not found — commit eval/datasets/hard_cases.jsonl separately)")


if __name__ == "__main__":
    main()
