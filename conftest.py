"""
conftest.py — pytest configuration for the UCI RAG Chatbot eval suite.

Adds the project root to sys.path so that `rag_chatbot` and `eval` are
importable when pytest is run from any directory.
"""

import sys
from pathlib import Path

# Project root = directory containing this conftest.py
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
