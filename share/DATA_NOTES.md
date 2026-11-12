# Experiment data notes (local Ollama run)

> **This is the local qwen2.5:14b run only. The report and slides will use the Colab run (GPT-4o-mini) instead.** The Colab numbers will differ, so do not mix the two. The Colab notebook's section 10 writes `per_question.csv` and `run_info.json` in the same format into Google Drive (`MyDrive/COMP4330_LightRAG/results_openai/`); use those for the final numbers. Keep this note as a backup and as a description of the local run.

Files in this folder: `per_question.csv` (92 rows = 46 questions x 2 systems) and this note.
Everything below comes from the local run with qwen2.5:14b. **No GPT-4o-mini / Colab run has been done yet**, so none of these numbers are from the model named in the proposal.

## Where the experiments stand

| Item | Status |
|---|---|
| Exp 1: naive RAG vs LightRAG (default `mix` mode) | **Done** locally, 46 questions, one run each |
| Exp 2: chunk size and top-k | **Not started** |
| Exp 3: the five query modes | **Not run on real data.** The code is in the notebook (section 9) and passed a 1-question test only |
| Improvement | Not started |
| Colab run with GPT-4o-mini | Not done |

Note: in the proposal, **Exp 3 is the comparison of query modes** (naive, local, global, hybrid, mix), not a multi-hop experiment. Multi-hop only shows up as a question type: 19 of the 46 questions come from multi-hop datasets (2WikiMQA 16, HotpotQA 3).

## Results (mean over 46 questions)

| System | Hit rate | nDCG | Faithfulness | Answer relevancy | Token F1 | Prompt tok/query | Completion tok/query | LLM calls/query | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| Naive RAG | 0.957 | 0.821 | 0.899 | 0.834 | 0.403 | 9,267 | 142 | 1 | 10.9 |
| LightRAG (mix) | 0.913 | 0.833 | 0.829 | 0.859 | 0.358 | 12,206 | 243 | 2 | 18.1 |

Token F1 wins per question: naive 33, LightRAG 13. Faithfulness: naive ahead on 14, LightRAG on 9, 23 ties.

## Corpus and indexing

* Source: UltraDomain `mix.jsonl`. We kept the shortest documents until about 100k tokens.
* **20 documents, 95,302 tokens (cl100k), 67,078 words, 94 chunks** (1200 tokens, 100 overlap).
* **46 questions:** qasper 19, 2WikiMQA 16, multifieldqa 8, HotpotQA 3.
* LightRAG index: **161 minutes, 156 LLM calls, 681,579 prompt + 334,531 completion tokens** (1.02M total). Resulting graph: 4,311 nodes, 2,652 edges.
* These index figures are for the **resumed run only**. The first attempt was stopped after it had processed 2 documents, and LightRAG reused its cached extraction output, so the true total is somewhat higher. Treat 161 min / 156 calls as a lower bound.
* Entity extraction was cut off by the output cap on many chunks. Per-document log lines such as "truncation hit 4 of 6 chunks" appear for all 20 documents (20 log lines). Summed over those lines this is roughly 60 of 94 chunks, but the log spans two runs, so do not quote an exact number.

## Model and settings

| Setting | Value |
|---|---|
| LLM (both systems, and judge) | qwen2.5:14b via Ollama, temperature 0, context 16,384 |
| Extraction output cap | `num_predict` 3000 tokens (qwen2.5 loops without it) |
| Embedding | bge-m3, 1024 dimensions |
| Chunking (both systems) | 1200 tokens, overlap 100 |
| Naive RAG | top-k = 8 chunks, cosine similarity, one LLM call |
| LightRAG | v1.5.7, mode `mix`, top_k 40 (entities/relations), chunk_top_k 20, max context 12,000 tokens, reranker off, 1 gleaning pass |
| Evaluation | RAGAS 0.4.3 faithfulness and answer relevancy, judge = same qwen2.5:14b |
| Retrieval metrics | Document level: a retrieved chunk is relevant if it comes from the question's source document |

## Columns in `per_question.csv`

`qid, question, system, mode, hit, ndcg, answer, gold_answer, token_f1, faithfulness, answer_relevancy, prompt_tokens, completion_tokens, llm_calls, latency_s, dataset, source_doc`

* **`judge_correct` does not exist.** We never ran a correctness judge, so there is nothing to put there. The closest columns are `token_f1` (overlap with the gold answer) and the two RAGAS scores. Do not describe any of them as "accuracy".
* `mode` is `naive_rag` or `lightrag_mix`. The five Exp 3 modes will add rows once that experiment has been run.
* `answer` is the raw model output (LightRAG answers still contain the trailing "References" list; it was stripped before scoring).

## Numbers we are not fully sure about (word these carefully)

1. **LightRAG faithfulness (0.829) is probably underestimated.** It was judged against the retrieved chunk text only, but LightRAG also gave the LLM entity and relation descriptions. The notebook is now fixed; the local numbers were not recomputed.
2. **Entity-level retrieval was often missing.** For 16 of 46 questions qwen2.5:14b returned no low-level keywords (high-level ones were always present), so for those questions LightRAG's mix run used relation-level retrieval plus chunk search only.
3. **Truncated extraction** means the knowledge graph is incomplete, which probably hurts LightRAG.
4. **n = 46, one run, no repeated trials.** The small differences (answer relevancy 0.859 vs 0.834, nDCG 0.833 vs 0.821) should not be called significant.
5. **Same model writes and judges the answers**, so RAGAS scores may be biased.
6. **Hit rate and nDCG are document level**, not passage level. Hit rate is high for both systems because the corpus has only 20 documents.
7. **Index cost is a lower bound** (see above).
