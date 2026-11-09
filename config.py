"""Shared settings so every experiment uses the same LLM, embedding and context budget."""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"              # raw downloads (mix.jsonl) are shared by every subset

# LIGHTRAG_SUBSET names a separate corpus/results pair (e.g. "smoke" for a quick test), so a small trial run
# never touches the real corpus or the resumable index; unset means the full experiment folders.
SUBSET = os.environ.get("LIGHTRAG_SUBSET", "").strip()
CORPUS = (DATA / SUBSET if SUBSET else DATA) / "corpus"          # one .txt per source document
QUESTIONS = (DATA / SUBSET if SUBSET else DATA) / "questions.jsonl"
RESULTS = ROOT / "results" / SUBSET if SUBSET else ROOT / "results"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")

# The experiment's models; these are always preferred when installed.
PRIMARY_LLM_MODEL = "qwen2.5:14b"
PRIMARY_JUDGE_MODEL = "qwen2.5-14b-judge"     # judge.Modelfile
# Local substitutes used only when the primary model is missing from Ollama.
FALLBACK_LLM_MODEL = "qwen3.5:9b-q4_K_M"
FALLBACK_JUDGE_MODEL = "qwen3.5-9b-judge"     # judge-qwen3.5-9b.Modelfile


# Returns the set of model names Ollama has installed, normalised to "name:tag", or None if Ollama
# cannot be reached; the short timeout keeps imports fast when the server is down.
def _installed_ollama_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as response:
            models = json.load(response).get("models", [])
    except (OSError, ValueError):
        return None
    return {name if ":" in name else f"{name}:latest" for name in (m.get("name", "") for m in models)}


# Chooses a model by precedence: an explicit environment variable always wins; otherwise the primary
# model if installed; otherwise the fallback if installed; otherwise the primary, so a missing model
# still fails loudly with its original name. Returns (model, used_fallback).
def _choose_model(env_name, primary, fallback, installed):
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        return explicit, False
    if installed is None:
        return primary, False

    # Ollama reports untagged models as ":latest", so both sides are compared in that form.
    def normalise(name):
        return name if ":" in name else f"{name}:latest"

    if normalise(primary) in installed:
        return primary, False
    if normalise(fallback) in installed:
        print(f"[config] {primary} is not installed in Ollama; using {fallback} instead.", file=sys.stderr)
        return fallback, True
    return primary, False


_INSTALLED = _installed_ollama_models()
LLM_MODEL, LLM_IS_FALLBACK = _choose_model("LIGHTRAG_LLM_MODEL", PRIMARY_LLM_MODEL, FALLBACK_LLM_MODEL, _INSTALLED)
JUDGE_MODEL, JUDGE_IS_FALLBACK = _choose_model("LIGHTRAG_JUDGE_MODEL", PRIMARY_JUDGE_MODEL, FALLBACK_JUDGE_MODEL,
                                               _INSTALLED)


# Reads LIGHTRAG_THINK as a tri-state switch for reasoning models (Ollama's think= / OpenAI's reasoning_effort).
# "0"/"false"/"no"/"off" return False and any other non-empty value returns True. When unset, an automatic
# Qwen 3.5 fallback gets False (its reasoning would otherwise use up the output-token cap), and every other
# model gets None, so the parameter is not sent and qwen2.5 behaves exactly as originally.
def _read_think_setting(is_fallback):
    value = os.environ.get("LIGHTRAG_THINK", "").strip()
    if value:
        return value.lower() not in {"0", "false", "no", "off"}
    return False if is_fallback else None


THINK = _read_think_setting(LLM_IS_FALLBACK)
JUDGE_THINK = _read_think_setting(JUDGE_IS_FALLBACK)
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
