"""Shared settings so every experiment uses the same LLM, embedding and context budget."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"          # one .txt per source document
QUESTIONS = DATA / "questions.jsonl"
RESULTS = ROOT / "results"

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"
EMBED_MODEL = "bge-m3:latest"
EMBED_DIM = 1024

# 16k context keeps a 14B model fully on a 16 GB GPU; LightRAG's default 30k budget would spill to CPU.
NUM_CTX = 16384
MAX_CONTEXT_TOKENS = 12000        # retrieved-context budget, same for both systems

CHUNK_SIZE = 1200                 # LightRAG default
CHUNK_OVERLAP = 100               # LightRAG default
NAIVE_TOP_K = 8                   # 8 x 1200 ~= 9.6k tokens, inside MAX_CONTEXT_TOKENS

# num_predict cap: qwen2.5:14b over-extracts and can loop until the context fills (>8 min per chunk).
LLM_OPTIONS = {"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 3000}
