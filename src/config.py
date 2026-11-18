"""Shared settings so every experiment uses the same LLM, embedding and context budget."""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repo root: scripts live in src/
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


### Judge backend. The RAGAS judge is a third-party model, DeepSeek, reached through the DeepSeek Harness (DSH)
### desktop app's command-line interface and its signed-in account (judge_setup.py), so the judge is no longer the
### model family that wrote the answers. If DSH is not installed, the local Ollama judge below is used instead, as
### before. Each judge writes its scores to its own folder (JUDGE_DIR), so cached scores from two judges never mix.
DSH_CLI = Path(os.environ.get("LIGHTRAG_DSH_CLI", "").strip() or Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
               / "DeepSeek Harness" / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd")
DSH_JUDGE_MODEL = os.environ.get("LIGHTRAG_DSH_MODEL", "").strip() or "deepseek-flash"
# None leaves DeepSeek's own default effort ("high"); DSH accepts off / low / high / max.
DSH_JUDGE_EFFORT = os.environ.get("LIGHTRAG_DSH_EFFORT", "").strip() or None


# Chooses "dsh" or "ollama" for the judge. LIGHTRAG_JUDGE_BACKEND always wins; otherwise DSH is used when its
# command-line launcher exists, and a missing DSH falls back to Ollama with a [config] notice on stderr.
def _choose_judge_backend():
    explicit = os.environ.get("LIGHTRAG_JUDGE_BACKEND", "").strip().lower()
    if explicit:
        if explicit not in {"dsh", "ollama"}:
            raise ValueError(f"LIGHTRAG_JUDGE_BACKEND must be 'dsh' or 'ollama', not {explicit!r}")
        return explicit
    if DSH_CLI.is_file():
        return "dsh"
    print(f"[config] DeepSeek Harness was not found at {DSH_CLI}; judging with the local Ollama judge instead.",
          file=sys.stderr)
    return "ollama"


JUDGE_BACKEND = _choose_judge_backend()

# How a DeepSeek judgement reaches DSH. "desktop": through the optional dsh-lightrag-judge plugin (kept outside this
# repository, in github.com/Test-Plus-XD/DSH-Discord-AutoReply under lightrag-judge/) inside the
# running desktop app, so each judgement appears live, thinking included, in this repository's DSH workspace.
# "headless": a separate `dsh --profile headless` run per prompt, saved to disk but not shown in the app.
DSH_PORT = int(os.environ.get("LIGHTRAG_DSH_PORT", "").strip() or 18791)


# Chooses the DSH transport. LIGHTRAG_DSH_TRANSPORT always wins; otherwise "desktop" when the plugin's API answers
# (the desktop app is open with the plugin installed), else "headless" with a [config] notice. Decide once per run
# and keep it: both use the same persona, prompts, model and effort, but DSH frames the request slightly differently.
def _choose_dsh_transport():
    explicit = os.environ.get("LIGHTRAG_DSH_TRANSPORT", "").strip().lower()
    if explicit:
        if explicit not in {"desktop", "headless"}:
            raise ValueError(f"LIGHTRAG_DSH_TRANSPORT must be 'desktop' or 'headless', not {explicit!r}")
        return explicit
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{DSH_PORT}/status", timeout=2) as response:
            if json.load(response).get("plugin") == "dsh-lightrag-judge":
                return "desktop"
    except (OSError, ValueError):
        pass
    # The plugin is optional (teammates may not have it), so this is information, not a warning.
    print(f"[config] Judging through headless DSH (the optional desktop plugin, dsh-lightrag-judge, is not answering on "
          f"port {DSH_PORT}, so judgements are not shown live in the app).", file=sys.stderr)
    return "headless"


DSH_TRANSPORT = _choose_dsh_transport() if JUDGE_BACKEND == "dsh" else None

_INSTALLED = _installed_ollama_models()
LLM_MODEL, LLM_IS_FALLBACK = _choose_model("LIGHTRAG_LLM_MODEL", PRIMARY_LLM_MODEL, FALLBACK_LLM_MODEL, _INSTALLED)
# The Ollama judge model is resolved only when Ollama judges, so a DSH run prints no misleading judge-fallback notice.
JUDGE_MODEL, JUDGE_IS_FALLBACK = (
    _choose_model("LIGHTRAG_JUDGE_MODEL", PRIMARY_JUDGE_MODEL, FALLBACK_JUDGE_MODEL, _INSTALLED)
    if JUDGE_BACKEND == "ollama" else (None, False))


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
# Only the Ollama judge takes this switch; DeepSeek's thinking is set through DSH_JUDGE_EFFORT instead.
JUDGE_THINK = _read_think_setting(JUDGE_IS_FALLBACK) if JUDGE_BACKEND == "ollama" else None


# Reads LIGHTRAG_JUDGE_REPEATS: how many independent judgings each RAGAS score averages (default 3). A smoke test
# showed one DeepSeek judging of the same answer can move faithfulness by about 0.15, so 3 to 5 judgings are the
# intended range; 1 restores the old single judging and is meant only for quick pipeline checks. The cap of 5 keeps a
# full rescoring (460 answers, about 1.5 min per judging through DSH) within reach.
def _read_judge_repeats():
    value = os.environ.get("LIGHTRAG_JUDGE_REPEATS", "").strip() or "3"
    if not value.isdigit() or not 1 <= int(value) <= 5:
        raise ValueError(f"LIGHTRAG_JUDGE_REPEATS must be a whole number from 1 to 5 (3 to 5 for real runs), "
                         f"not {value!r}")
    return int(value)


JUDGE_REPEATS = _read_judge_repeats()

# A readable judge name for logs and share/run_info.json, and the folder that holds this judge's score caches
# (scores.jsonl, exp2/exp3_scores.jsonl) and summaries. The Ollama judge keeps the original location, RESULTS
# itself, so the existing local scores stay where they are; a DeepSeek judge gets its own subfolder per model and
# effort, because a different model or effort is a different judge. The number of judgings is not part of the
# folder: every cached score keeps its individual judgings, so a run with more repeats only adds the missing ones.
# JUDGE_MODEL_ID is the short judge name written into every score record and exported file (judge_model column),
# beside the answering model (rag_model), so each result says which model answered and which one judged.
if JUDGE_BACKEND == "dsh":
    JUDGE_NAME = (f"{DSH_JUDGE_MODEL} via DeepSeek Harness ({DSH_TRANSPORT}, account), "
                  f"effort {DSH_JUDGE_EFFORT or 'default (high)'}")
    JUDGE_DIR = RESULTS / ("judge_" + DSH_JUDGE_MODEL + (f"_{DSH_JUDGE_EFFORT}" if DSH_JUDGE_EFFORT else ""))
    JUDGE_MODEL_ID = DSH_JUDGE_MODEL + (f" (effort {DSH_JUDGE_EFFORT})" if DSH_JUDGE_EFFORT else "")
else:
    JUDGE_NAME = f"{JUDGE_MODEL} via Ollama"
    JUDGE_DIR = RESULTS
    JUDGE_MODEL_ID = JUDGE_MODEL
JUDGE_NAME += f", mean of {JUDGE_REPEATS} judging{'s' if JUDGE_REPEATS > 1 else ''}"
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
