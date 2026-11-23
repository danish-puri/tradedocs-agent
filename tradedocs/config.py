"""Paths, model settings, and the paper trading URL."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The only Alpaca host this project will ever call. It is hard-coded on
# purpose so a typo in .env can't point the agent at a live account.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"

MODEL = os.getenv("TRADEDOCS_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("TRADEDOCS_EFFORT", "medium")

SOURCES_FILE = ROOT / "sources.txt"
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
ENDPOINTS_FILE = DATA_DIR / "endpoints.json"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "alpaca_docs"

# How many doc chunks to hand the model for each question.
TOP_K = 6
# Chunk size in characters. The local embedding model only reads about the
# first 256 tokens of each chunk, so chunks are kept close to that.
CHUNK_CHARS = 1000

# Most API calls the agent may propose for one question.
MAX_CALLS = 4


def read_sources(path: Path = SOURCES_FILE) -> list[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
