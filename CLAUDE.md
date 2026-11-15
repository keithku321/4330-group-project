# CLAUDE.md

COMP 4330SEF P04 Group 5 project: evaluating LightRAG against a naive RAG baseline on a subset of UltraDomain (`mix.jsonl`). Answers and embeddings run locally through Ollama; the RAGAS judge is DeepSeek through DeepSeek Harness (see Models). The group proposal plans three experiments (1: naive RAG vs LightRAG; 2: chunk size and top-k; 3: the five query modes naive/local/global/hybrid/mix) plus one improvement (retrieval weighting, extraction-prompt tuning, or reranking). All three experiments are implemented and have been run locally (`exp1.py` + `evaluate.py`, `exp2.py`, `exp3.py`); the improvement is not started. `export.py` collects the results into `share/` (described in `share/DATA_NOTES.md`). `COMP4330_Group5_LightRAG.ipynb` is the Colab (Gemini/GPT) version; never mix its numbers with the local run. `PLAN.md` is the group's project plan.

## Environment

- Python 3.14 virtual environment in `.venv`, built from `requirements.txt`, which deliberately uses `--pre` and the scientific-python nightly index. Expect dev builds (numpy, pandas, pyarrow); run `.venv\Scripts\python -m pip check` after changing it.
- Keep `langchain-community<0.4`: ragas 0.4.x still imports `ChatVertexAI`, removed in 0.4.
- Always run scripts with `.venv\Scripts\python -X utf8 ...` (Windows console encoding).
- `LightRAG.pyproj` / `LightRAG.slnx` are the Visual Studio project. Every tracked file must be listed there by hand, including files that arrive through a pull or merge from teammates: `Compile` for `.py`, `Content` for everything else (Markdown, CSV, JSON, notebooks, Modelfiles). Keep each item group alphabetical, write subfolder paths with backslashes (`share\run_info.json`), and give every new subfolder a `<Folder Include="share\" />` item. Preserve its CRLF line endings. After a pull, compare `git ls-files` against the `Include=` entries to catch anything missing.

## Models (all decided in `config.py`)

- Primary models are the experiment's originals and must not be replaced: LLM `qwen2.5:14b`, embeddings `bge-m3:latest` (1024-dim, must match `EMBED_DIM`).
- If the primary LLM is not installed, `config.py` falls back automatically to `qwen3.5:9b-q4_K_M` and prints a `[config]` notice to stderr.
- The RAGAS judge is a third-party model (group decision, 15-Nov-2026), so it no longer shares a model family with the answers: DeepSeek (`deepseek-flash` by default) through the DeepSeek Harness (DSH) desktop app's CLI on its signed-in account, not the DeepSeek API (`judge_setup.py`, overlay `dsh-judge.patch.yml`). The user reads every judgement as a DSH session in the desktop app, so keep judging inside DSH; do not switch it to the DeepSeek HTTP API. Two transports (`LIGHTRAG_DSH_TRANSPORT`): `desktop` posts to the DSH plugin `dsh-lightrag-judge`, which is NOT part of this repository: it lives in the user's own repo https://github.com/Test-Plus-XD/DSH-Discord-AutoReply under `lightrag-judge/` (checked out at `C:\Users\Test-Plus\.dsh\profiles\desktop\plugins\discord-autoreply`, linked into `~/.dsh/profiles/desktop`, local API `127.0.0.1:18791`); never add plugin code to this group repo, which creates each session live in the repo's DSH workspace; `headless` runs `dsh --profile headless` and the app does not list those sessions. Default is `desktop` when the plugin answers, else `headless`. The plugin is optional (teammates may not have it, or DSH at all): never make it a requirement, and keep the headless and Ollama paths working. Plugin changes take effect only after restarting DeepSeek Harness. Save the plugin's files as UTF-8 **without a BOM** (never Windows PowerShell's `Set-Content -Encoding utf8`): a BOM in `package.json` makes DSH skip the whole bundle at boot without any visible error, so `/status` on port 18791 simply stops answering. If DSH is not installed, the judge falls back to Ollama's `qwen2.5-14b-judge` (`judge.Modelfile`), or `qwen3.5-9b-judge` (`judge-qwen3.5-9b.Modelfile`) when that is missing, with a `[config]` notice.
- Environment overrides: `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_JUDGE_MODEL` (Ollama judge; always win), `LIGHTRAG_THINK` (`0`/`1`), `LIGHTRAG_JUDGE_BACKEND` (`dsh`/`ollama`), `LIGHTRAG_DSH_MODEL` (`deepseek-flash`/`deepseek-v4-pro`), `LIGHTRAG_DSH_EFFORT` (`off`/`low`/`high`/`max`), `LIGHTRAG_DSH_TRANSPORT` (`desktop`/`headless`), `LIGHTRAG_DSH_PORT` (plugin API, `18791`), `LIGHTRAG_DSH_CLI` (launcher path).
- Each judge keeps its scores in its own folder, `config.JUDGE_DIR`: `results/<subset>/` for the Ollama judge (the original location), `results/<subset>/judge_<model>[_<effort>]/` for DeepSeek. Never copy or merge score files between those folders, and run `export.py` with the same judge settings as the scoring run. A DSH sign-in or quota error stops scoring (`JudgeUnavailableError`) rather than caching empty scores.
- Qwen 3.5 is a thinking model. Thinking must be off for it, or the hidden reasoning consumes `num_predict`/`max_tokens` and truncates extraction. The native API takes `think=False` (`naive_rag.py`, and LightRAG via a `"think"` key in `options`); the OpenAI-compatible `/v1` API used by RAGAS takes `reasoning_effort="none"` instead. Nothing is sent for qwen2.5, which does not support thinking.
- Do not pull or create Ollama models without asking the user.

## Data and results

- `LIGHTRAG_SUBSET=<name>` redirects the corpus to `data/<name>/` and results to `results/<name>/`, so test runs never touch the real corpus or the resumable index. `data/mix.jsonl` is shared.
- Quick end-to-end check (about 27 min with the 9B fallback and the Ollama judge: index ~13 min for 5 chunks, answers ~3 min, judging ~10 min; the DeepSeek judge takes about 1.5 min per answer, 5 DSH calls). Never rerun `exp1.py index` on a finished subset: it overwrites `exp1_index_cost.json` with a near-zero cost; use a new subset name instead:
  ```
  $env:LIGHTRAG_SUBSET='smoke'
  .venv\Scripts\python -X utf8 prep_data.py --target-tokens 5000
  .venv\Scripts\python -X utf8 exp1.py index
  .venv\Scripts\python -X utf8 exp1.py run --limit 2
  .venv\Scripts\python -X utf8 evaluate.py results\smoke\exp1_naive.jsonl results\smoke\exp1_lightrag.jsonl
  ```
- Delete `results/<subset>/` before re-indexing after a model change: LightRAG's extraction cache (`enable_llm_cache_for_entity_extract`) would otherwise reuse the old model's output.
- `data/` and `results/` are git-ignored; `share/` is committed and is fully regenerated by `export.py`, so edit the scripts, not the files in it (`share/DATA_NOTES.md` is the one hand-written file). With `LIGHTRAG_SUBSET` set, `export.py` writes to `results/<subset>/share/` instead.
- Exp 2 and Exp 3 build on Exp 1's results folder. Order: `exp1.py index` → `exp3.py run` → `exp3.py score` → `exp2.py index` → `exp2.py run` → `exp2.py score` → `export.py`. `exp2.py score` reuses Exp 3's `mix` scores (from the same judge folder) for its default chunk 1200 / top-k 40 cell, and every script builds its judge with `judge_setup.make_judges()`.

## Fairness rules for experiments

Both systems must share the same corpus, questions, LLM, embedding model, chunking (1200/100) and context budget (`MAX_CONTEXT_TOKENS`); change settings in `config.py`, never in one system only. Every system in a comparison must be scored by the same judge (same `JUDGE_DIR`). Do not run `evaluate.py` while indexing: both use the same GPU and Ollama swaps models (the DeepSeek judge still embeds with local bge-m3).
