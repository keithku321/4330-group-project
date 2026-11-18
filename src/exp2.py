"""Experiment 2 (local): LightRAG chunk size x top-k (2x2) in the default `mix` mode.

    python -X utf8 src/exp2.py index           # build the chunk-600 graph -> results/lightrag_index_c600 (slow, resumable)
    python -X utf8 src/exp2.py run [--limit N] # answer every setting -> results/exp2_c<chunk>_k<top_k>.jsonl (resumable)
    python -X utf8 src/exp2.py score           # judge + metrics -> <judge dir>/exp2_scores.jsonl, exp2_summary.csv (resumable)

Chunk 1200 / top-k 40 is LightRAG's default and is exactly the Exp 3 `mix` run, so its answers and scores are reused
(the scores from the same judge's exp3_scores.jsonl; <judge dir> is config.JUDGE_DIR, see judge_setup.py).
top_k is how many entities (low level) and relations (high level) LightRAG retrieves; chunk_top_k stays at 20.
Run the steps one after another: indexing, answering and judging all use the same GPU.
"""
import argparse
import asyncio
import collections
import csv
import json
import time

from lightrag import QueryParam

from config import CHUNK_SIZE, CORPUS, JUDGE_DIR, LLM_MODEL, QUESTIONS, RESULTS
from evaluate import model_columns
from exp3 import METRICS, answer, append, judges, load_jsonl, score_record
from judge_setup import judging_complete, save_score
from lightrag_setup import TRACKER, make_rag

SETTINGS = [(600, 20), (600, 40), (1200, 20), (1200, 40)]
DEFAULT = (CHUNK_SIZE, QueryParam().top_k)
SCORES = JUDGE_DIR / "exp2_scores.jsonl"
assert DEFAULT in SETTINGS, DEFAULT


def index_dir(chunk_size):
    return RESULTS / ("lightrag_index" if chunk_size == CHUNK_SIZE else f"lightrag_index_c{chunk_size}")


def answers_path(chunk_size, top_k):
    return RESULTS / ("exp3_mix.jsonl" if (chunk_size, top_k) == DEFAULT else f"exp2_c{chunk_size}_k{top_k}.jsonl")


def system(chunk_size, top_k):
    return f"lightrag-mix-c{chunk_size}-k{top_k}"


async def index():
    docs = sorted(CORPUS.glob("*.txt"))
    for cs in sorted({cs for cs, _ in SETTINGS} - {CHUNK_SIZE}):
        cost = RESULTS / f"exp2_index_cost_c{cs}.json"
        if cost.exists():  # a rerun of a finished index would overwrite the real cost with ~0
            continue
        rag = await make_rag(index_dir(cs), chunk_size=cs)
        TRACKER.reset()
        t0 = time.perf_counter()
        await rag.ainsert([p.read_text(encoding="utf-8") for p in docs], ids=[p.stem for p in docs],
                          file_paths=[p.name for p in docs])
        stats = {"chunk_size": cs, "docs": len(docs), "index_time_s": round(time.perf_counter() - t0, 1),
                 **TRACKER.get_usage()}
        cost.write_text(json.dumps(stats, indent=2))
        print("index cost:", stats, flush=True)
        await rag.finalize_storages()


async def run(limit):
    questions = load_jsonl(QUESTIONS)[:limit]
    for cs in sorted({cs for cs, _ in SETTINGS}, reverse=True):
        if not (RESULTS / f"exp2_index_cost_c{cs}.json").exists() and cs != CHUNK_SIZE:
            print(f"chunk {cs}: index not finished, skipped")
            continue
        rag = await make_rag(index_dir(cs), chunk_size=cs)
        for k in sorted(k for c, k in SETTINGS if c == cs and (c, k) != DEFAULT):
            out = answers_path(cs, k)
            done = {r["qid"] for r in load_jsonl(out)}
            for i, q in enumerate(questions):
                if q["qid"] in done:
                    continue
                res = await answer(rag, q["question"], "mix", top_k=k)
                append(out, {**q, "system": system(cs, k), "chunk_size": cs, "top_k": k,
                             "rag_model": LLM_MODEL, **res})
                print(f"[c{cs} k{k}] {i + 1}/{len(questions)} {q['qid']} {res['latency_s']}s", flush=True)
        await rag.finalize_storages()


async def score():
    faith, relevancy = judges()
    reused = {s["qid"]: s for s in load_jsonl(JUDGE_DIR / "exp3_scores.jsonl") if s["system"] == "lightrag-mix"}
    done = {(s["chunk_size"], s["top_k"], s["qid"]): s for s in load_jsonl(SCORES)}
    for cs, k in SETTINGS:
        records = load_jsonl(answers_path(cs, k))
        if not records:
            continue
        # nDCG's ideal ranking needs each doc's chunk count at this chunk size (at 1200 it equals naive_index's)
        kv = json.loads((index_dir(cs) / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
        n_chunks = collections.Counter(c["file_path"].removesuffix(".txt") for c in kv.values())
        for i, rec in enumerate(records):
            previous = done.get((cs, k, rec["qid"]))
            if (cs, k) == DEFAULT:
                # The default cell mirrors Exp 3's mix score; it is copied again whenever that score has changed
                # (for example after Exp 3 was topped up with more judgings), so the two never disagree.
                s = {**reused[rec["qid"]], "system": system(cs, k), "chunk_size": cs, "top_k": k}
                if s == previous:
                    continue
            elif judging_complete(previous):
                continue
            else:
                s = {**await score_record(rec, faith, relevancy, n_chunks, previous), "system": system(cs, k),
                     "chunk_size": cs, "top_k": k}
            save_score(SCORES, s, ("chunk_size", "top_k", "qid"))
            print(f"[c{cs} k{k}] {i + 1}/{len(records)} faith={s['faithfulness']} rel={s['answer_relevancy']}",
                  flush=True)
    summarize()


def summarize():
    scores = load_jsonl(SCORES)
    rows = []
    for cs, k in SETTINGS:
        group = [s for s in scores if (s["chunk_size"], s["top_k"]) == (cs, k)]
        row = {"chunk_size": cs, "top_k": k, "n": len(group), **model_columns(group)}
        for m in METRICS:
            vals = [s[m] for s in group if s.get(m) is not None]
            row[m] = round(sum(vals) / len(vals), 3) if vals else None
        rows.append(row)
    cols = ["chunk_size", "top_k", "n", "rag_model", "judge_model", *METRICS]
    with (JUDGE_DIR / "exp2_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("| " + " | ".join(cols) + " |")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "run", "score"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    asyncio.run({"index": index, "score": score}.get(args.step, lambda: run(args.limit))())
