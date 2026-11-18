# Exploration and Evaluation of LightRAG: A Graph-Enhanced Retrieval-Augmented Generation Framework

COMP 4330SEF Advanced Programming and AI Algorithms, 2026 Autumn, P04 Group 5.

This repository holds the code and the local results for the three experiments in our proposal: naive RAG vs LightRAG (Exp 1), chunk size and top-k (Exp 2), and LightRAG's five query modes (Exp 3). `notebooks/COMP4330_Group5_LightRAG.ipynb` is the Google Colab version (GPT-4o-mini, as in the proposal); the `.py` scripts are the local version (qwen2.5:14b through Ollama). The two runs use different models, so do not mix their numbers.

The collected local results are in [`share/`](share): `per_question.csv` (one row per question and system), `run_info.json` (models, settings, corpus, index cost), the per-experiment summaries and `DATA_NOTES.md`.

## Project layout

```
.
├── src/                  Python scripts (run from the repo root: .venv\Scripts\python -X utf8 src\exp1.py ...)
│   ├── config.py         shared settings: models, paths, chunking, context budget
│   ├── prep_data.py      builds the UltraDomain subset in data/
│   ├── naive_rag.py      naive RAG baseline
│   ├── lightrag_setup.py LightRAG on Ollama, with token tracking
│   ├── judge_setup.py    RAGAS judge (DeepSeek through DSH, or Ollama)
│   ├── evaluate.py       scoring for Exp 1
│   ├── exp1.py exp2.py exp3.py   the three experiments
│   └── export.py         collects results into share/
├── configs/              Ollama judge Modelfiles and the DSH judge overlay
├── notebooks/            Google Colab version of the whole pipeline
├── docs/                 project plan and handover notes
├── share/                exported results for the report (CSV, JSON, DATA_NOTES.md)
├── data/                 corpus and questions (mostly git-ignored; prep_data.py rebuilds it)
├── results/              answers, scores, indexes and logs (large files git-ignored)
├── requirements.txt
└── LightRAG.slnx / LightRAG.pyproj   Visual Studio project
```

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
.venv\Scripts\python -X utf8 src\prep_data.py --target-tokens 5000
.venv\Scripts\python -X utf8 src\exp1.py index
.venv\Scripts\python -X utf8 src\exp1.py run --limit 2
.venv\Scripts\python -X utf8 src\evaluate.py results\smoke\exp1_naive.jsonl results\smoke\exp1_lightrag.jsonl
Remove-Item Env:LIGHTRAG_SUBSET
```

With the 9B fallback this took about 27 minutes in total: indexing 5 chunks took about 13 minutes, answering 2 questions with each system about 3 minutes, and judging about 10 minutes with the local judge, one judging per score (about 6 minutes per round of judgings with the DeepSeek judge, so about 18 minutes at the default 3 judgings; see "Smoke test of the DeepSeek judge" below). Even with thinking off, 2 of the 5 extraction responses reached the `num_predict` cap (see "Capped extraction output" below).

If you change the LLM, delete `results\<subset>\` before indexing again. Otherwise LightRAG's extraction cache reuses the old model's output.

## Experiment 1: naive RAG vs LightRAG

```
.venv\Scripts\python -X utf8 src\exp1.py index   # about 3 hours; can resume after interruption
.venv\Scripts\python -X utf8 src\exp1.py run     # writes results/exp1_naive.jsonl and results/exp1_lightrag.jsonl
```

Both systems use the same settings:

- **LLM:** qwen2.5:14b
- **Embedding:** bge-m3
- **Chunking:** 1200 tokens with 100 overlap
- **Context budget:** about 12k tokens

LightRAG runs with its default `mix` mode and reranking turned off. Every answer is saved with its token counts, latency and the retrieved document ids.

## Evaluation

```
.venv\Scripts\python -X utf8 src\evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
```

### The judge: DeepSeek through DeepSeek Harness

**The problem.** Our first runs had two judge problems, both raised in the progress slides ("The faithfulness judge does not repeat itself"):

1. **Self-judgement.** qwen2.5:14b both wrote the answers and judged them, so a judge could favour its own model's style.
2. **No repeatability.** A repeat run showed that the judge is not even consistent with itself: 42 of 46 answers came out word for word the same, yet faithfulness differed on 22 of those 42 (mean |Δ| 0.12).

**The fix.** The judging now addresses both problems:

1. **A third-party judge, DeepSeek**, from a different model family than the qwen models that write the answers, which removes self-judgement entirely.
2. **The mean of several judgings per score** (3 by default, up to 5; see "Every score is the mean of several judgings" below), so no score rests on one roll of the dice. Each score also records its spread, so the remaining judge noise is measured and reported rather than hidden.

The smoke test below shows why both parts are needed: DeepSeek alone was still not repeatable from one judging to the next, while the mean of three judgings varies about 1.7 times less.

DeepSeek is reached through the DeepSeek Harness (DSH) desktop app rather than the DeepSeek API, on the DeepSeek **account** signed in there (no API key; usage counts against that account's quota). Every RAGAS prompt becomes one fresh, tool-less DSH session with a judge persona, titled by system, question and metric (e.g. `naive q030 faithfulness`). There are two routes (`LIGHTRAG_DSH_TRANSPORT`):

- **`desktop` (optional, watch live).** Nobody needs this to run the experiments; it only adds live viewing. The DSH plugin `dsh-lightrag-judge` is kept outside this repository, in [`lightrag-judge/` of DSH-Discord-AutoReply](https://github.com/Test-Plus-XD/DSH-Discord-AutoReply/tree/main/lightrag-judge). It runs inside the open desktop app and serves a local API on `127.0.0.1:18791`. `judge_setup.py` posts each prompt there, and the plugin creates the session in this repository's DSH workspace, so you can watch every judgement, DeepSeek's thinking included, appear and stream in the sidebar under **LightRAG** while scoring runs. Clone that repository, install the plugin once into the desktop profile, then restart DeepSeek Harness:
  ```
  & "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd" plugin --profile desktop add "link:<path to DSH-Discord-AutoReply>/lightrag-judge"
  ```
- **`headless` (the default without the plugin).** Each prompt runs as `dsh --profile headless` with the overlay `configs/dsh-judge.patch.yml`, which re-enables the account sign-in, removes every tool and keeps the repository's `CLAUDE.md` out of the context. These sessions are saved on disk but the desktop app does not list them.

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
| `LIGHTRAG_JUDGE_REPEATS` | how many independent judgings each score averages: `3` (default) to `5`; `1` only for quick pipeline checks. Applies to both judges |

**Every score is the mean of several judgings.** One judging of the same answer is too noisy to rely on (see the smoke test below), so each faithfulness and answer-relevancy score is the mean of `LIGHTRAG_JUDGE_REPEATS` independent judgings, each in a fresh judge session titled e.g. `naive q030 faithfulness 2/3`. The score caches keep every judging (`faithfulness_runs`) beside the mean, its sample standard deviation (`faithfulness_sd`) and the repeat count (`judge_repeats`), and likewise for answer relevancy. Scoring is resumable judging by judging: a rerun, or a run with a higher repeat count, adds only the judgings still missing (a score cached before this change counts as its first judging), and a judging that failed is tried again on the next run. The mean always uses the first `LIGHTRAG_JUDGE_REPEATS` judgings, so every answer in a comparison is averaged over the same number. The cost grows with the count: at the default of 3, one answer takes about 4.5 minutes through DSH, so a full rescoring of the 460 answers is a long, resumable job.

**Every result names its models.** Each answer record stores the model that wrote it (`rag_model`), and each score record stores both that model and the judge (`judge_model`, e.g. `deepseek-flash` or `qwen3.5-9b-judge`). The summaries, `per_question.csv` and `run_info.json` carry the same two columns, so a file can never be mistaken for another model's results. Answers written before this change have no `rag_model`, so the currently configured LLM is assumed for them, as `run_info.json` always has; score them with the models that produced them. Judgings by a different judge model are never averaged together: a cached record from another judge is judged again from scratch.

**Each judge keeps its own scores.** The Ollama judge writes to `results\` as before; a DeepSeek judge writes to `results\judge_<model>[_<effort>]\` (`scores.jsonl`, `summary.csv`, `exp2_scores.jsonl`, `exp3_scores.jsonl` and their summaries). The caches therefore never mix, and both judges' scores of the same answers can be compared. `export.py` reads the folder of the judge that is currently configured, so run it with the same settings as the scoring run.

If DSH reports that the account is not signed in or out of quota, scoring stops with a `JudgeUnavailableError` instead of recording empty scores; sign in through the desktop app and rerun (finished answers are skipped).

### Smoke test of the DeepSeek judge (15-Nov-2026)

The whole pipeline was run on a fresh 5k-token subset (`LIGHTRAG_SUBSET=dsh-smoke`: 2 documents, 2 questions answered) with the 9B fallback LLM and DeepSeek `deepseek-flash` (headless route, default effort). Every step finished without a failed judge call, and `export.py` wrote to `results\dsh-smoke\share\`, leaving the committed `share/` untouched.

| Step | Time |
|---|---|
| `exp1.py index` (2 documents) | 11.5 min |
| `exp1.py run --limit 2` | 3 min |
| `evaluate.py`, DeepSeek judge (4 answers) | 5.8 min |
| `exp3.py run --limit 2` / `exp3.py score` (10 answers) | 8 min / 14.9 min |
| `exp2.py score` (reuses the Exp 3 `mix` scores) + `export.py` | 0.7 min |

The same four Exp 1 answers were then judged a second time by DeepSeek, in new DSH sessions, and once by the local Ollama judge (`qwen3.5-9b-judge`). Faithfulness:

| Answer | DeepSeek, 1st judging | DeepSeek, 2nd judging | Local Ollama judge |
|---|---|---|---|
| naive q030 | 0.944 | 0.667 | 1.000 |
| naive q031 | 0.900 | 0.667 | 1.000 |
| LightRAG q030 | 0.792 | 0.818 | 1.000 |
| LightRAG q031 | 0.750 | 0.833 | 0.864 |

DeepSeek is much stricter than the local judge, which gave three of the four answers a perfect score. It is not repeatable either: between its two judgings faithfulness moved by 0.155 on average (answer relevancy by 0.032), and the naive-vs-LightRAG order reversed (means 0.922 / 0.771, then 0.667 / 0.826). Four answers are far too few to measure this properly, but they are enough to show that changing the judge alone does not remove judge noise, which is why every score is now the mean of several judgings (see above).

**With three judgings per answer.** The same four answers were then topped up from one judging (the 1st judging above) to three, through the desktop route; the two new judgings took 6 minutes for all four answers. Faithfulness:

| Answer | Judgings | Mean | Spread (sd) |
|---|---|---|---|
| naive q030 | 0.944, 0.818, 0.889 | 0.884 | 0.063 |
| naive q031 | 0.900, 1.000, 0.923 | 0.941 | 0.052 |
| LightRAG q030 | 0.792, 0.789, 0.947 | 0.843 | 0.091 |
| LightRAG q031 | 0.750, 0.733, 0.850 | 0.778 | 0.063 |

The typical spread of a single judging is 0.067 for faithfulness and 0.017 for answer relevancy, so the mean of three is roughly $\sqrt{3}\approx1.7$ times steadier (about 0.04). The system means are naive 0.912 and LightRAG 0.810. The second judging in the table above (0.667 for both naive answers) is not part of these means: it was a separate experiment kept in its own file, and the top-up only builds on the live score cache. Exp 3 was topped up the same way (10 answers, about 13 minutes in all). Its mean faithfulness per mode with three judgings, against the earlier single judging:

| Mode | naive | local | global | hybrid | mix |
|---|---|---|---|---|---|
| 1 judging | 0.703 | 0.796 | 0.804 | 0.865 | 0.883 |
| Mean of 3 judgings | 0.806 | 0.855 | 0.811 | 0.873 | 0.870 |

The single judging had put `naive` far behind every graph mode (0.703); with three judgings the gap shrinks to about 0.05 to 0.07, which is the kind of change one noisy judging can cause. Over all 14 smoke answers, a single judging's faithfulness spread averages 0.06 (answer relevancy 0.016). The desktop plugin stopped answering partway through, so the two `mix` answers were finished on the headless route; both routes send the same prompt, persona, model and effort, but a real run should stay on one route. Afterwards `exp2.py score` re-copied the updated `mix` scores into its default cell by itself, and `export.py` reported no score left to rescore, with `qwen3.5:9b-q4_K_M` as `rag_model` and `deepseek-flash` as `judge_model` on every row.

### The local Ollama judge (fallback)

```
ollama create qwen2.5-14b-judge -f configs\1
$env:LIGHTRAG_JUDGE_BACKEND = 'ollama'
.venv\Scripts\python -X utf8 src\evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
```

When only the 9B model is available, create its judge instead with `ollama create qwen3.5-9b-judge -f configs\1`. The judge model `qwen2.5-14b-judge` is qwen2.5:14b with a 16k context set in its Modelfile. Ollama's OpenAI-compatible API, which RAGAS uses, cannot set the context size per request, so it would otherwise fall back to Ollama's 4k default.

The evaluation packages are already in `requirements.txt`. `langchain-community` has to stay below 0.4, because ragas 0.4.3 still imports `ChatVertexAI`, which was removed in 0.4.

**Metrics**

| Group | Metric | How it is measured |
|---|---|---|
| Retrieval | hit rate, nDCG | At document level: a retrieved chunk counts as relevant if it comes from the question's source document |
| Generation | faithfulness, answer relevancy | RAGAS, with DeepSeek (or the local fallback) as judge |
| Generation | token F1 | Compared against the gold answer |
| Cost | tokens, LLM calls, latency | Copied from the run |

Per-answer scores go to `scores.jsonl` in the judge's folder, and the averages for each system to `summary.csv` next to it, with the answering model and the judge in its `rag_model` and `judge_model` columns. The numbers in `share/` were produced before the fix, by a single judging from the local qwen2.5:14b judge, which is the same model that wrote the answers; they have to be rescored with the DeepSeek judge (averaged judgings) before they are compared with any DeepSeek-judged result.

Do not run the evaluation while indexing is in progress. The embeddings (and the Ollama judge, if used) need the same GPU, so Ollama keeps swapping the models in and out of memory.

## Experiment 2: chunk size and top-k

```
.venv\Scripts\python -X utf8 src\exp2.py index   # builds a second graph with 600-token chunks; several hours, resumable
.venv\Scripts\python -X utf8 src\exp2.py run
.venv\Scripts\python -X utf8 src\exp2.py score   # writes exp2_scores.jsonl and exp2_summary.csv in the judge's folder
```

A 2x2 grid in `mix` mode: chunk size 600 / 1200 x top-k 20 / 40. top-k is how many entities and relations LightRAG retrieves; `chunk_top_k` stays at its default of 20. Chunk 1200 / top-k 40 is LightRAG's default, which is the Exp 3 `mix` run, so it is reused rather than rerun.

## Experiment 3: query modes

```
.venv\Scripts\python -X utf8 src\exp3.py run     # naive, local, global, hybrid, mix on the Exp 1 index
.venv\Scripts\python -X utf8 src\exp3.py score   # writes exp3_scores.jsonl and exp3_summary.csv in the judge's folder
```

Faithfulness here is judged against everything LightRAG gave the LLM: the chunk text and the entity and relation descriptions.

## Export

```
.venv\Scripts\python -X utf8 src\export.py       # writes share/per_question.csv, share/run_info.json and the summaries
```

`export.py` takes the scores from the configured judge's folder and records that judge in `run_info.json`. Every row of `per_question.csv` names the model that answered (`rag_model`) and the one that judged (`judge_model`), and gives each RAGAS score's spread and number of judgings (`faithfulness_sd`, `faithfulness_n`, `answer_relevancy_sd`, `answer_relevancy_n`); `run_info.json` lists the models found (`rag_models`, `judge_models`), the repeat count and how many scores still need rescoring, which the export also prints as an `[export]` warning. With `LIGHTRAG_SUBSET` set it writes to `results\<subset>\share\` instead, so a test run never overwrites the committed `share/`.

## Differences from the proposal

- **Local models instead of an API.** We use local qwen2.5:14b and bge-m3 instead of GPT-4o-mini and an embedding API. This keeps the cost at zero.
- **Smaller context window.** Context is set to 16k tokens instead of LightRAG's 30k default, so that the model fits entirely on a 16 GB GPU.
- **Capped extraction output.** Extraction output is limited with `num_predict=3000`, because qwen2.5 otherwise over-extracts and loops. This cap truncates some chunks, so a few relations are lost.
