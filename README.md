# Exploration and Evaluation of LightRAG: A Graph-Enhanced Retrieval-Augmented Generation Framework

COMP 4330SEF Advanced Programming and AI Algorithms, 2026 Autumn, P04 Group 5.

This repository holds the code and the local results for the three experiments in our proposal: naive RAG vs LightRAG (Exp 1), chunk size and top-k (Exp 2), and LightRAG's five query modes (Exp 3). `COMP4330_Group5_LightRAG.ipynb` is the Google Colab version (GPT-4o-mini, as in the proposal); the `.py` scripts are the local version (qwen2.5:14b through Ollama). The two runs use different models, so do not mix their numbers.

The collected local results are in [`share/`](share): `per_question.csv` (one row per question and system), `run_info.json` (models, settings, corpus, index cost), the per-experiment summaries and `DATA_NOTES.md`.

## Setup

The experiments run locally on an RTX 5060 Ti (16 GB) through Ollama, so there are no API costs. Judging goes through the DeepSeek account signed in to the DeepSeek Harness desktop app (see [Evaluation](#evaluation)), so it uses that account's quota rather than a paid API key.

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama pull bge-m3
```

`requirements.txt` targets Python 3.14 and allows pre-release and nightly builds (`--pre`, plus the scientific-python nightly index for numpy). It covers both the experiment and the evaluation packages. The Visual Studio project is `LightRAG.slnx`, and it uses the `.venv` interpreter.

### Models

`config.py` chooses the models when it is imported:

| Role | Preferred | Used automatically if the preferred one is not available |
|---|---|---|
| LLM | `qwen2.5:14b` | `qwen3.5:9b-q4_K_M` |
| RAGAS judge | DeepSeek `deepseek-flash` through DeepSeek Harness (see [Evaluation](#evaluation)) | `qwen2.5-14b-judge` (`judge.Modelfile`), then `qwen3.5-9b-judge` (`judge-qwen3.5-9b.Modelfile`) |
| Embeddings | `bge-m3` | none |

A fallback is announced on stderr with a `[config]` line. Results produced with the 9B model are not directly comparable to those from the 14B model, and scores from different judges are never comparable. Qwen 3.5 is a thinking model, so its thinking is switched off for both answering and local judging; otherwise the hidden reasoning uses up the output-token cap. You can override the choice with `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_JUDGE_MODEL` (the Ollama judge), `LIGHTRAG_THINK` (`0` or `1`) and the judge variables listed under [Evaluation](#evaluation).

## Data

`prep_data.py` takes a subset of UltraDomain `mix.jsonl`. It keeps the shortest documents until the total reaches about 100k tokens. The result is 20 documents (95k tokens) and their 46 questions. Of the 46 questions, 19 are multi-hop (2WikiMQA 16, HotpotQA 3); the rest are single-document (qasper 19, multifieldqa 8).

### Quick test on a tiny subset

Setting `LIGHTRAG_SUBSET` sends the corpus to `data/<name>/` and the results to `results/<name>/`. A trial run therefore never touches the real corpus or the resumable index. With a 5k-token budget the subset is the 2 shortest documents (about 4 chunks) and their 5 questions:

```
$env:LIGHTRAG_SUBSET = 'smoke'
.venv\Scripts\python -X utf8 prep_data.py --target-tokens 5000
.venv\Scripts\python -X utf8 exp1.py index
.venv\Scripts\python -X utf8 exp1.py run --limit 2
.venv\Scripts\python -X utf8 evaluate.py results\smoke\exp1_naive.jsonl results\smoke\exp1_lightrag.jsonl
Remove-Item Env:LIGHTRAG_SUBSET
```

With the 9B fallback this took about 27 minutes in total: indexing 5 chunks took about 13 minutes, answering 2 questions with each system about 3 minutes, and judging about 10 minutes. Even with thinking off, 2 of the 5 extraction responses reached the `num_predict` cap (see "Capped extraction output" below).

If you change the LLM, delete `results\<subset>\` before indexing again. Otherwise LightRAG's extraction cache reuses the old model's output.

## Experiment 1: naive RAG vs LightRAG

```
.venv\Scripts\python -X utf8 exp1.py index   # about 3 hours; can resume after interruption
.venv\Scripts\python -X utf8 exp1.py run     # writes results/exp1_naive.jsonl and results/exp1_lightrag.jsonl
```

Both systems use the same settings:

- **LLM:** qwen2.5:14b
- **Embedding:** bge-m3
- **Chunking:** 1200 tokens with 100 overlap
- **Context budget:** about 12k tokens

LightRAG runs with its default `mix` mode and reranking turned off. Every answer is saved with its token counts, latency and the retrieved document ids.

## Evaluation

```
.venv\Scripts\python -X utf8 evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
```

### The judge: DeepSeek through DeepSeek Harness

Our first runs used qwen2.5:14b both to write the answers and to judge them, and a repeat run showed that this judge is not even consistent with itself: 42 of 46 answers came out word for word the same, yet faithfulness differed on 22 of those 42. We therefore switched to a third-party judge, **DeepSeek**. It is reached through the DeepSeek Harness (DSH) desktop app rather than the DeepSeek API, on the DeepSeek **account** signed in there (no API key; usage counts against that account's quota). Every RAGAS prompt becomes one fresh, tool-less DSH session with a judge persona, titled by system, question and metric (e.g. `naive q030 faithfulness`). There are two routes (`LIGHTRAG_DSH_TRANSPORT`):

- **`desktop` (optional, watch live).** Nobody needs this to run the experiments; it only adds live viewing. The DSH plugin `dsh-lightrag-judge` is kept outside this repository, in [`lightrag-judge/` of DSH-Discord-AutoReply](https://github.com/Test-Plus-XD/DSH-Discord-AutoReply/tree/main/lightrag-judge). It runs inside the open desktop app and serves a local API on `127.0.0.1:18791`. `judge_setup.py` posts each prompt there, and the plugin creates the session in this repository's DSH workspace, so you can watch every judgement, DeepSeek's thinking included, appear and stream in the sidebar under **LightRAG** while scoring runs. Clone that repository, install the plugin once into the desktop profile, then restart DeepSeek Harness:
  ```
  & "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd" plugin --profile desktop add "link:<path to DSH-Discord-AutoReply>/lightrag-judge"
  ```
- **`headless` (the default without the plugin).** Each prompt runs as `dsh --profile headless` with the overlay `dsh-judge.patch.yml`, which re-enables the account sign-in, removes every tool and keeps the repository's `CLAUDE.md` out of the context. These sessions are saved on disk but the desktop app does not list them.

By default `desktop` is used when the plugin answers and `headless` otherwise (with an informational `[config]` line); without DeepSeek Harness at all, the local Ollama judge is used, so a teammate needs neither DSH nor the plugin to run anything. Both routes send the same prompt, persona, model and effort, but keep one route for a whole experiment; the open app must stay open for `desktop`, and closing it stops scoring with a clear error (rerun to resume).

Requirements: install DeepSeek Harness and sign in to DeepSeek once in the desktop app. Each RAGAS call becomes one fresh DSH session (no judgement sees another), and the reply must be the JSON instance of the metric's schema, the same instruction `instructor` gives the Ollama judge; an unparseable reply is retried up to 3 times. One answer takes 5 DSH calls (2 for faithfulness, 3 for answer relevancy), about 1.5 minutes in total. The answer-relevancy embeddings still use local bge-m3. DSH does not expose a temperature for headless runs, so DeepSeek runs with its own sampling defaults.

| Variable | Effect |
|---|---|
| `LIGHTRAG_JUDGE_BACKEND` | `dsh` or `ollama`. Default: `dsh` when DeepSeek Harness is installed, otherwise `ollama` with a `[config]` notice |
| `LIGHTRAG_DSH_MODEL` | `deepseek-flash` (default) or `deepseek-v4-pro` |
| `LIGHTRAG_DSH_EFFORT` | `off`, `low`, `high` or `max` reasoning; unset uses DeepSeek's default (`high`) |
| `LIGHTRAG_DSH_TRANSPORT` | `desktop` (through the plugin, visible live in the app) or `headless`. Default: `desktop` when the plugin answers |
| `LIGHTRAG_DSH_PORT` | the plugin's local API port, default `18791` (must match the plugin's `apiPort`) |
| `LIGHTRAG_DSH_CLI` | path to `dsh.cmd`, if DeepSeek Harness is not installed in `%LOCALAPPDATA%\Programs\DeepSeek Harness` |

**Each judge keeps its own scores.** The Ollama judge writes to `results\` as before; a DeepSeek judge writes to `results\judge_<model>[_<effort>]\` (`scores.jsonl`, `summary.csv`, `exp2_scores.jsonl`, `exp3_scores.jsonl` and their summaries). The caches therefore never mix, and both judges' scores of the same answers can be compared. `export.py` reads the folder of the judge that is currently configured, so run it with the same settings as the scoring run.

If DSH reports that the account is not signed in or out of quota, scoring stops with a `JudgeUnavailableError` instead of recording empty scores; sign in through the desktop app and rerun (finished answers are skipped).

### The local Ollama judge (fallback)

```
ollama create qwen2.5-14b-judge -f judge.Modelfile
$env:LIGHTRAG_JUDGE_BACKEND = 'ollama'
.venv\Scripts\python -X utf8 evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
```

When only the 9B model is available, create its judge instead with `ollama create qwen3.5-9b-judge -f judge-qwen3.5-9b.Modelfile`. The judge model `qwen2.5-14b-judge` is qwen2.5:14b with a 16k context set in its Modelfile. Ollama's OpenAI-compatible API, which RAGAS uses, cannot set the context size per request, so it would otherwise fall back to Ollama's 4k default.

The evaluation packages are already in `requirements.txt`. `langchain-community` has to stay below 0.4, because ragas 0.4.3 still imports `ChatVertexAI`, which was removed in 0.4.

**Metrics**

| Group | Metric | How it is measured |
|---|---|---|
| Retrieval | hit rate, nDCG | At document level: a retrieved chunk counts as relevant if it comes from the question's source document |
| Generation | faithfulness, answer relevancy | RAGAS, with DeepSeek (or the local fallback) as judge |
| Generation | token F1 | Compared against the gold answer |
| Cost | tokens, LLM calls, latency | Copied from the run |

Per-answer scores go to `scores.jsonl` in the judge's folder, and the averages for each system to `summary.csv` next to it. The numbers in `share/` were produced before the switch, by the local qwen2.5:14b judge, which is the same model that wrote the answers; they have to be rescored with the DeepSeek judge before they are compared with any DeepSeek-judged result.

Do not run the evaluation while indexing is in progress. The embeddings (and the Ollama judge, if used) need the same GPU, so Ollama keeps swapping the models in and out of memory.

## Experiment 2: chunk size and top-k

```
.venv\Scripts\python -X utf8 exp2.py index   # builds a second graph with 600-token chunks; several hours, resumable
.venv\Scripts\python -X utf8 exp2.py run
.venv\Scripts\python -X utf8 exp2.py score   # writes exp2_scores.jsonl and exp2_summary.csv in the judge's folder
```

A 2x2 grid in `mix` mode: chunk size 600 / 1200 x top-k 20 / 40. top-k is how many entities and relations LightRAG retrieves; `chunk_top_k` stays at its default of 20. Chunk 1200 / top-k 40 is LightRAG's default, which is the Exp 3 `mix` run, so it is reused rather than rerun.

## Experiment 3: query modes

```
.venv\Scripts\python -X utf8 exp3.py run     # naive, local, global, hybrid, mix on the Exp 1 index
.venv\Scripts\python -X utf8 exp3.py score   # writes exp3_scores.jsonl and exp3_summary.csv in the judge's folder
```

Faithfulness here is judged against everything LightRAG gave the LLM: the chunk text and the entity and relation descriptions.

## Export

```
.venv\Scripts\python -X utf8 export.py       # writes share/per_question.csv, share/run_info.json and the summaries
```

`export.py` takes the scores from the configured judge's folder and records that judge in `run_info.json`. With `LIGHTRAG_SUBSET` set it writes to `results\<subset>\share\` instead, so a test run never overwrites the committed `share/`.

## Differences from the proposal

- **Local models instead of an API.** We use local qwen2.5:14b and bge-m3 instead of GPT-4o-mini and an embedding API. This keeps the cost at zero.
- **Smaller context window.** Context is set to 16k tokens instead of LightRAG's 30k default, so that the model fits entirely on a 16 GB GPU.
- **Capped extraction output.** Extraction output is limited with `num_predict=3000`, because qwen2.5 otherwise over-extracts and loops. This cap truncates some chunks, so a few relations are lost.
