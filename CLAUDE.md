# CLAUDE.md

COMP 4330SEF P04 Group 5 project: evaluating LightRAG against a naive RAG baseline on a subset of UltraDomain (`mix.jsonl`), fully local through Ollama. The group proposal plans three experiments (1: naive RAG vs LightRAG; 2: chunk size and top-k; 3: the five query modes naive/local/global/hybrid/mix) plus one improvement (retrieval weighting, extraction-prompt tuning, or reranking). Only Experiment 1 is implemented so far (`exp1.py`).

## Environment

- Python 3.14 virtual environment in `.venv`, built from `requirements.txt`, which deliberately uses `--pre` and the scientific-python nightly index. Expect dev builds (numpy, pandas, pyarrow); run `.venv\Scripts\python -m pip check` after changing it.
- Keep `langchain-community<0.4`: ragas 0.4.x still imports `ChatVertexAI`, removed in 0.4.
- Always run scripts with `.venv\Scripts\python -X utf8 ...` (Windows console encoding).
- `LightRAG.pyproj` / `LightRAG.slnx` are the Visual Studio project. Every new file must be added there by hand (`Compile` for `.py`, `Content` for everything else); preserve its CRLF line endings.

## Models (all decided in `config.py`)

- Primary models are the experiment's originals and must not be replaced: LLM `qwen2.5:14b`, judge `qwen2.5-14b-judge` (`judge.Modelfile`), embeddings `bge-m3:latest` (1024-dim, must match `EMBED_DIM`).
- If a primary model is not installed, `config.py` falls back automatically to `qwen3.5:9b-q4_K_M` / `qwen3.5-9b-judge` (`judge-qwen3.5-9b.Modelfile`) and prints a `[config]` notice to stderr.
- Environment overrides: `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_JUDGE_MODEL` (always win), `LIGHTRAG_THINK` (`0`/`1`).
- Qwen 3.5 is a thinking model. Thinking must be off for it, or the hidden reasoning consumes `num_predict`/`max_tokens` and truncates extraction. The native API takes `think=False` (`naive_rag.py`, and LightRAG via a `"think"` key in `options`); the OpenAI-compatible `/v1` API used by RAGAS takes `reasoning_effort="none"` instead. Nothing is sent for qwen2.5, which does not support thinking.
- Do not pull or create Ollama models without asking the user.

## Data and results

- `LIGHTRAG_SUBSET=<name>` redirects the corpus to `data/<name>/` and results to `results/<name>/`, so test runs never touch the real corpus or the resumable index. `data/mix.jsonl` is shared.
- Quick end-to-end check (about 27 min with the 9B fallback: index ~13 min for 5 chunks, answers ~3 min, judging ~10 min):
  ```
  $env:LIGHTRAG_SUBSET='smoke'
  .venv\Scripts\python -X utf8 prep_data.py --target-tokens 5000
  .venv\Scripts\python -X utf8 exp1.py index
  .venv\Scripts\python -X utf8 exp1.py run --limit 2
  .venv\Scripts\python -X utf8 evaluate.py results\smoke\exp1_naive.jsonl results\smoke\exp1_lightrag.jsonl
  ```
- Delete `results/<subset>/` before re-indexing after a model change: LightRAG's extraction cache (`enable_llm_cache_for_entity_extract`) would otherwise reuse the old model's output.
- `data/` and `results/` are git-ignored.

## Fairness rules for experiments

Both systems must share the same corpus, questions, LLM, embedding model, chunking (1200/100) and context budget (`MAX_CONTEXT_TOKENS`); change settings in `config.py`, never in one system only. Do not run `evaluate.py` while indexing: both use the same GPU and Ollama swaps models.
