"""
Diagnostic script — run from project root with venv active:
    python diagnose.py
"""
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/embeddings")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)

# ── Majors collection — CS major check ────────────────────────────────────────
print("=" * 60)
print("MAJORS DB — Computer Science major lookup")
print("=" * 60)
majors_client = chromadb.PersistentClient(path="data/db/majors")
majors_col = majors_client.get_collection("majors")
print(f"Total docs in majors collection: {majors_col.count()}")

# Find any docs whose URL contains 'computerscience' or 'informationandcomputer'
print("\n--- Scanning all majors IDs for ICS school pages (limit 5000) ---")
all_major_results = majors_col.get(limit=5000, include=["metadatas"])

# Precise check: pages actually from the ICS school URL path
ics_school_docs = [
    m for m in all_major_results["metadatas"]
    if m and "donaldbrenschoolofinformationandcomputersciences" in (m.get("url") or "").lower()
]
print(f"ICS school pages found: {len(ics_school_docs)}")

# Check for the CS B.S. page specifically
cs_bs_docs = [
    m for m in ics_school_docs
    if "computerscience_bs" in (m.get("url") or "").lower()
       or "computerscience/computerscience_b" in (m.get("url") or "").lower()
]
print(f"  CS B.S. pages found: {len(cs_bs_docs)}")
for m in cs_bs_docs[:10]:
    print(f"    {m.get('url', '?')}")

# Show all unique ICS school URLs so we can see exactly what was crawled
unique_ics_urls = sorted({m.get("url", "") for m in ics_school_docs})
print(f"\n  All unique ICS school URLs ({len(unique_ics_urls)} pages):")
for url in unique_ics_urls[:40]:
    print(f"    {url}")

# Semantic query against majors
print("\n--- Semantic query: 'Computer Science major required math courses' ---")
query = "Computer Science major required math courses"
embedding = fn([query])[0]
results = majors_col.query(query_embeddings=[embedding], n_results=10, include=["documents", "metadatas"])
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    url = (meta or {}).get("url", "?")
    print(f"  {url[:80]} | {doc[:80]}")

# ── Courses collection ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COURSES DB")
print("=" * 60)
client = chromadb.PersistentClient(path="data/db/courses")
col = client.get_collection("courses")
print(f"Total docs in courses collection: {col.count()}")

# Check if MATH 2A exists by ID
math2a = col.get(ids=["MATH 2A"], include=["documents", "metadatas"])
if math2a["documents"]:
    print(f"\nMATH 2A found by ID:\n{math2a['documents'][0][:300]}")
else:
    print("\nMATH 2A NOT found by ID.")

math2b = col.get(ids=["MATH 2B"], include=["documents"])
if math2b["documents"]:
    print(f"\nMATH 2B found by ID:\n{math2b['documents'][0][:300]}")
else:
    print("\nMATH 2B NOT found by ID.")

# Validate the catalog codes that students commonly abbreviate
print("\n--- Checking commonly-abbreviated course IDs ---")
for course_id in ["COMPSCI 161", "I&C SCI 6B", "I&C SCI 6D"]:
    r = col.get(ids=[course_id], include=["documents", "metadatas"])
    if r["documents"]:
        code_meta = (r["metadatas"][0] or {}).get("code", "?")
        print(f"  FOUND   {course_id!r:20s} (metadata code={code_meta!r}): {r['documents'][0][:80]}")
    else:
        print(f"  MISSING {course_id!r:20s} — check crawl/ingest output")
