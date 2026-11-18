"""Experiment 3 (local): LightRAG query modes naive / local / global / hybrid / mix on the existing index.

    python -X utf8 src/exp3.py run [--limit N]   # answer with every mode -> results/exp3_<mode>.jsonl (resumable)
    python -X utf8 src/exp3.py score             # judge + metrics -> <judge dir>/exp3_scores.jsonl, exp3_summary.csv (resumable)

Same index, questions, LLM and settings as Exp 1. `mix` is rerun here (not copied from Exp 1) so that every mode's
faithfulness is judged against the same kind of context: chunk text AND entity/relation descriptions.
Run `run` first and `score` afterwards: the answer model and the judge's embeddings (and the Ollama judge, if used)
must not share the GPU at the same time. <judge dir> is config.JUDGE_DIR (see judge_setup.py).
"""
import argparse
import asyncio
import collections
import csv
import json
import time

from lightrag import QueryParam
from lightrag.operate import get_keywords_from_query

from config import JUDGE_DIR, LLM_MODEL, MAX_CONTEXT_TOKENS, QUESTIONS, RESULTS
from evaluate import model_columns, ndcg, rag_model, strip_references, token_f1
from judge_setup import judge_answer, judging_complete, make_judges, save_score
from lightrag_setup import TRACKER, make_rag

MODES = ["naive", "local", "global", "hybrid", "mix"]
SCORES = JUDGE_DIR / "exp3_scores.jsonl"
METRICS = ["hit", "ndcg", "faithfulness", "answer_relevancy", "token_f1", "n_entities", "n_relations",
           "prompt_tokens", "completion_tokens", "llm_calls", "latency_s"]


def load_jsonl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def append(path, rec):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def answer(rag, question, mode, **param):
    TRACKER.reset()
    t0 = time.perf_counter()
    r = await rag.aquery_llm(question, param=QueryParam(
        mode=mode, response_type="Single Paragraph", enable_rerank=False, max_total_tokens=MAX_CONTEXT_TOKENS, **param))
    latency = time.perf_counter() - t0
    data, usage = r.get("data") or {}, TRACKER.get_usage()
    chunks = data.get("chunks", [])
    kg_text = [f"{e.get('entity_name')}: {e.get('description')}" for e in data.get("entities", [])] + \
              [f"{x.get('src_id')} -> {x.get('tgt_id')}: {x.get('description')}" for x in data.get("relationships", [])]
    return {
        "answer": (r.get("llm_response") or {}).get("content", ""),
        "retrieved_doc_ids": [c.get("file_path", "").removesuffix(".txt") for c in chunks],
        "contexts": [c.get("content", "") for c in chunks] + kg_text,
        "n_entities": len(data.get("entities", [])),
        "n_relations": len(data.get("relationships", [])),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "llm_calls": usage["call_count"],
        "latency_s": round(latency, 2),
    }


async def run(limit):
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")][:limit]
    rag = await make_rag()
    diag = RESULTS / "exp3_keyword_diagnostic.json"
    if not diag.exists() and limit is None:   # a --limit test run must not save a partial diagnostic
        gc = rag._build_global_config()
        kws = [await get_keywords_from_query(q["question"], QueryParam(mode="mix"), gc, rag.llm_response_cache)
               for q in questions]
        diag.write_text(json.dumps({"questions": len(kws), "empty_low_level": sum(not ll for hl, ll in kws),
                                    "empty_high_level": sum(not hl for hl, ll in kws)}, indent=2))
        print("keyword diagnostic:", diag.read_text())
    for mode in MODES:
        out = RESULTS / f"exp3_{mode}.jsonl"
        done = {r["qid"] for r in load_jsonl(out)}
        for i, q in enumerate(questions):
            if q["qid"] in done:
                continue
            append(out, {**q, "system": f"lightrag-{mode}", "rag_model": LLM_MODEL,
                         **await answer(rag, q["question"], mode)})
            print(f"[{mode}] {i + 1}/{len(questions)} {q['qid']}", flush=True)
    await rag.finalize_storages()


# Builds the RAGAS faithfulness and answer-relevancy scorers (shared with exp2.py). The judge itself, DeepSeek through
# DeepSeek Harness or the local Ollama judge, is chosen in config.py and built by judge_setup.make_judges, the same
# function evaluate.py uses, so all three experiments are scored by one judge.
def judges():
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    return make_judges()


# Scores one answer: retrieval and token-F1 metrics locally, then both RAGAS metrics, each the mean of JUDGE_REPEATS
# judgings (judge_setup.judge_answer), topping up the judgings already cached in previous. An unusable judge
# (JudgeUnavailableError) stops the run; any other judge failure loses only that judging.
async def score_record(rec, faith, relevancy, n_chunks, previous=None):
    ans = strip_references(rec["answer"])
    return {"system": rec["system"], "qid": rec["qid"], "rag_model": rag_model(rec),
            "hit": float(rec["doc_id"] in rec["retrieved_doc_ids"]),
            "ndcg": ndcg(rec["retrieved_doc_ids"], rec["doc_id"], n_chunks[rec["doc_id"]]),
            "token_f1": token_f1(ans, rec["answers"]),
            **{k: rec[k] for k in ("n_entities", "n_relations", "prompt_tokens", "completion_tokens",
                                   "llm_calls", "latency_s")},
            **await judge_answer(rec["question"], ans, rec["contexts"], faith, relevancy,
                                 f"{rec['system']} {rec['qid']}", previous)}


async def score():
    chunks = json.loads((RESULTS / "naive_index" / "chunks.json").read_text(encoding="utf-8"))
    n_chunks = collections.Counter(c["doc_id"] for c in chunks)
    faith, relevancy = judges()
    done = {(s["system"], s["qid"]): s for s in load_jsonl(SCORES)}
    for mode in MODES:
        records = load_jsonl(RESULTS / f"exp3_{mode}.jsonl")
        for i, rec in enumerate(records):
            previous = done.get((rec["system"], rec["qid"]))
            if judging_complete(previous):
                continue
            s = await score_record(rec, faith, relevancy, n_chunks, previous)
            save_score(SCORES, s, ("system", "qid"))
            print(f"[{rec['system']}] {i + 1}/{len(records)} faith={s['faithfulness']} rel={s['answer_relevancy']}",
                  flush=True)
    summarize()


def summarize():
    scores = load_jsonl(SCORES)
    rows = []
    for mode in MODES:
        group = [s for s in scores if s["system"] == f"lightrag-{mode}"]
        row = {"system": f"lightrag-{mode}", "n": len(group), **model_columns(group)}
        for m in METRICS:
            vals = [s[m] for s in group if s.get(m) is not None]
            row[m] = round(sum(vals) / len(vals), 3) if vals else None
        rows.append(row)
    with (JUDGE_DIR / "exp3_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "n", "rag_model", "judge_model", *METRICS])
        w.writeheader()
        w.writerows(rows)
    print("| " + " | ".join(["system", "n", "rag_model", "judge_model", *METRICS]) + " |")
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in ["system", "n", "rag_model", "judge_model", *METRICS]) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    asyncio.run(run(args.limit) if args.step == "run" else score())
