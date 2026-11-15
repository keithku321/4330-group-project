"""Experiment 1: naive RAG vs LightRAG (default mix mode) on the same corpus, questions and LLM.

    python exp1.py index            # build the LightRAG graph (slow, resumable) and the naive vector index
    python exp1.py run [--limit N]  # answer the questions with both systems -> results/exp1_*.jsonl (resumable)
"""
import argparse
import asyncio
import json
import time

from lightrag import QueryParam

from config import CORPUS, LLM_MODEL, MAX_CONTEXT_TOKENS, QUESTIONS, RESULTS
from lightrag_setup import TRACKER, make_rag
from naive_rag import NaiveRAG


def load_done(path):
    return {json.loads(l)["qid"] for l in path.open(encoding="utf-8")} if path.exists() else set()


async def index():
    docs = sorted(CORPUS.glob("*.txt"))
    NaiveRAG()  # builds and saves the vector index on first use
    rag = await make_rag()
    TRACKER.reset()
    t0 = time.perf_counter()
    await rag.ainsert([p.read_text(encoding="utf-8") for p in docs], ids=[p.stem for p in docs],
                      file_paths=[p.name for p in docs])
    stats = {"docs": len(docs), "index_time_s": round(time.perf_counter() - t0, 1), **TRACKER.get_usage()}
    (RESULTS / "exp1_index_cost.json").write_text(json.dumps(stats, indent=2))
    print("LightRAG index cost:", stats)
    await rag.finalize_storages()


async def lightrag_answer(rag, question):
    TRACKER.reset()
    t0 = time.perf_counter()
    r = await rag.aquery_llm(question, param=QueryParam(
        mode="mix", response_type="Single Paragraph", enable_rerank=False, max_total_tokens=MAX_CONTEXT_TOKENS))
    latency = time.perf_counter() - t0
    data = r.get("data") or {}
    chunks = data.get("chunks", [])
    usage = TRACKER.get_usage()
    return {
        "answer": (r.get("llm_response") or {}).get("content", ""),
        "retrieved_doc_ids": [c.get("file_path", "").removesuffix(".txt") for c in chunks],
        "contexts": [c.get("content", "") for c in chunks],
        "n_entities": len(data.get("entities", [])),
        "n_relations": len(data.get("relationships", [])),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "llm_calls": usage["call_count"],  # keyword extraction + answer
        "latency_s": round(latency, 2),
    }


async def run(limit):
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")][:limit]
    naive, rag = NaiveRAG(), await make_rag()
    for name, answer in [("naive", lambda q: asyncio.to_thread(naive.query, q)),
                         ("lightrag", lambda q: lightrag_answer(rag, q))]:
        out = RESULTS / f"exp1_{name}.jsonl"
        done = load_done(out)
        for i, q in enumerate(questions):
            if q["qid"] in done:
                continue
            res = await answer(q["question"])
            with out.open("a", encoding="utf-8") as f:
                # rag_model names the model that wrote this answer, so later files can show it beside the judge
                f.write(json.dumps({**q, "system": name, "rag_model": LLM_MODEL, **res}, ensure_ascii=False) + "\n")
            print(f"[{name}] {i + 1}/{len(questions)} {q['qid']} {res['latency_s']}s")
    await rag.finalize_storages()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "run"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)  # parents: a LIGHTRAG_SUBSET run nests under results/
    asyncio.run(index() if args.step == "index" else run(args.limit))
