# Exploration and Evaluation of LightRAG: A Graph-Enhanced Retrieval-Augmented Generation Framework

COMP 4330SEF Advanced Programming and AI Algorithms, 2026 Autumn, P04 Group 5.

This repository holds the code and the local results for the three experiments in our proposal: naive RAG vs LightRAG (Exp 1), chunk size and top-k (Exp 2), and LightRAG's five query modes (Exp 3). `COMP4330_Group5_LightRAG.ipynb` is the Google Colab version (GPT-4o-mini, as in the proposal); the `.py` scripts are the local version (qwen2.5:14b through Ollama). The two runs use different models, so do not mix their numbers.

The collected local results are in [`share/`](share): `per_question.csv` (one row per question and system), `run_info.json` (models, settings, corpus, index cost), the per-experiment summaries and `DATA_NOTES.md`.

## Setup

The experiments run locally on an RTX 5060 Ti (16 GB) through Ollama, so there are no API costs.

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama pull bge-m3
```

`requirements.txt` targets Python 3.14 and allows pre-release and nightly builds (`--pre`, plus the scientific-python nightly index for numpy). It covers both the experiment and the evaluation packages. The Visual Studio project is `LightRAG.slnx`, and it uses the `.venv` interpreter.

### Models

`config.py` chooses the models when it is imported:

| Role | Preferred (the experiment's model) | Used automatically if the preferred one is not installed |
|---|---|---|
| LLM | `qwen2.5:14b` | `qwen3.5:9b-q4_K_M` |
| RAGAS judge | `qwen2.5-14b-judge` (`judge.Modelfile`) | `qwen3.5-9b-judge` (`judge-qwen3.5-9b.Modelfile`) |
| Embeddings | `bge-m3` | none |

A fallback is announced on stderr with a `[config]` line. Results produced with the 9B model are not directly comparable to those from the 14B model. Qwen 3.5 is a thinking model, so its thinking is switched off for both answering and judging; otherwise the hidden reasoning uses up the output-token cap. You can override the choice with `LIGHTRAG_LLM_MODEL`, `LIGHTRAG_JUDGE_MODEL` and `LIGHTRAG_THINK` (`0` or `1`).

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
ollama create qwen2.5-14b-judge -f judge.Modelfile
.venv\Scripts\python -X utf8 evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
```

The evaluation packages are already in `requirements.txt`. When only the 9B model is available, create its judge instead with `ollama create qwen3.5-9b-judge -f judge-qwen3.5-9b.Modelfile`.

`langchain-community` has to stay below 0.4, because ragas 0.4.3 still imports `ChatVertexAI`, which was removed in 0.4. The judge model `qwen2.5-14b-judge` is qwen2.5:14b with a 16k context set in its Modelfile. Ollama's OpenAI-compatible API, which RAGAS uses, cannot set the context size per request, so it would otherwise fall back to Ollama's 4k default.

**Metrics**

| Group | Metric | How it is measured |
|---|---|---|
| Retrieval | hit rate, nDCG | At document level: a retrieved chunk counts as relevant if it comes from the question's source document |
| Generation | faithfulness, answer relevancy | RAGAS, with the local LLM as judge |
| Generation | token F1 | Compared against the gold answer |
| Cost | tokens, LLM calls, latency | Copied from the run |

Per-answer scores go to `results/scores.jsonl`. The averages for each system go to `results/summary.csv`. The judge is the same model that generates the answers; this is a known source of bias and should be listed under limitations.

Do not run the evaluation while indexing is in progress. The judge needs the same GPU, so Ollama keeps swapping the two models in and out of memory.

## Experiment 2: chunk size and top-k

```
.venv\Scripts\python -X utf8 exp2.py index   # builds a second graph with 600-token chunks; several hours, resumable
.venv\Scripts\python -X utf8 exp2.py run
.venv\Scripts\python -X utf8 exp2.py score   # writes results/exp2_scores.jsonl and results/exp2_summary.csv
```

A 2x2 grid in `mix` mode: chunk size 600 / 1200 x top-k 20 / 40. top-k is how many entities and relations LightRAG retrieves; `chunk_top_k` stays at its default of 20. Chunk 1200 / top-k 40 is LightRAG's default, which is the Exp 3 `mix` run, so it is reused rather than rerun.

## Experiment 3: query modes

```
.venv\Scripts\python -X utf8 exp3.py run     # naive, local, global, hybrid, mix on the Exp 1 index
.venv\Scripts\python -X utf8 exp3.py score   # writes results/exp3_scores.jsonl and results/exp3_summary.csv
```

Faithfulness here is judged against everything LightRAG gave the LLM: the chunk text and the entity and relation descriptions.

## Export

```
.venv\Scripts\python -X utf8 export.py       # writes share/per_question.csv, share/run_info.json and the summaries
```

## Differences from the proposal

- **Local models instead of an API.** We use local qwen2.5:14b and bge-m3 instead of GPT-4o-mini and an embedding API. This keeps the cost at zero.
- **Smaller context window.** Context is set to 16k tokens instead of LightRAG's 30k default, so that the model fits entirely on a 16 GB GPU.
- **Capped extraction output.** Extraction output is limited with `num_predict=3000`, because qwen2.5 otherwise over-extracts and loops. This cap truncates some chunks, so a few relations are lost.
