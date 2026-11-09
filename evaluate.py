"""Score experiment answers.

    python -X utf8 evaluate.py results/exp1_naive.jsonl results/exp1_lightrag.jsonl [--limit N]

Per answer (cached in results/scores.jsonl, so reruns only score new answers):
  retrieval  hit rate, nDCG        doc-level: a retrieved chunk is relevant if it comes from the question's source doc
  generation faithfulness, answer relevancy   RAGAS 0.4, local qwen2.5:14b judge + bge-m3
             token F1 vs gold answer          SQuAD-style, cheap sanity check
  cost       prompt/completion tokens, LLM calls, latency (copied from the run)
Prints a per-system summary and writes results/summary.csv.
"""
import argparse
import asyncio
import collections
import csv
import json
import math
import re
import string

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from config import OLLAMA_HOST, RESULTS

JUDGE_MODEL = "qwen2.5-14b-judge"   # qwen2.5:14b with num_ctx 16384 baked in (judge.Modelfile)
SCORES = RESULTS / "scores.jsonl"


def strip_references(answer):
    """LightRAG appends a '### References' list; it is not part of the answer's claims."""
    return re.split(r"\n#*\s*References\s*\n", answer)[0].strip()


def tokens(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return [w for w in text.split() if w not in {"a", "an", "the"}]


def token_f1(pred, golds):
    best = 0.0
    for gold in golds:
        common = sum((collections.Counter(tokens(pred)) & collections.Counter(tokens(gold))).values())
        if common:
            p, r = common / len(tokens(pred)), common / len(tokens(gold))
            best = max(best, 2 * p * r / (p + r))
    return best


def ndcg(retrieved_doc_ids, gold_doc, n_gold_chunks):
    rels = [d == gold_doc for d in retrieved_doc_ids]
    dcg = sum(1 / math.log2(i + 2) for i, rel in enumerate(rels) if rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rels), n_gold_chunks)))
    return dcg / ideal if ideal else 0.0


async def ragas_scores(rec, faith, relevancy):
    out = {}
    for name, coro in [("faithfulness", faith.ascore(user_input=rec["question"], response=rec["answer"],
                                                     retrieved_contexts=rec["contexts"] or [""])),
                       ("answer_relevancy", relevancy.ascore(user_input=rec["question"], response=rec["answer"]))]:
        try:
            out[name] = (await coro).value
        except Exception as e:  # a malformed judge reply loses one score, not the whole run
            print(f"  {name} failed for {rec['system']}/{rec['qid']}: {str(e)[:150]}")
            out[name] = None
    return out


async def main(files, limit):
    chunks = json.loads((RESULTS / "naive_index" / "chunks.json").read_text(encoding="utf-8"))
    n_chunks = collections.Counter(c["doc_id"] for c in chunks)

    client = AsyncOpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")  # Ollama ignores the key
    llm = llm_factory(JUDGE_MODEL, client=client, temperature=0, max_tokens=3000)
    emb = embedding_factory("openai", "bge-m3", client=client)
    faith, relevancy = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb)

    done = {}
    if SCORES.exists():
        for line in SCORES.open(encoding="utf-8"):
            s = json.loads(line)
            done[(s["file"], s["qid"])] = s

    for path in files:
        records = [json.loads(l) for l in open(path, encoding="utf-8")][:limit]
        for i, rec in enumerate(records):
            if (path, rec["qid"]) in done:
                continue
            rec["answer"] = strip_references(rec["answer"])
            score = {
                "file": path, "qid": rec["qid"], "system": rec["system"],
                "hit": float(rec["doc_id"] in rec["retrieved_doc_ids"]),
                "ndcg": ndcg(rec["retrieved_doc_ids"], rec["doc_id"], n_chunks[rec["doc_id"]]),
                "token_f1": token_f1(rec["answer"], rec["answers"]),
                **await ragas_scores(rec, faith, relevancy),
                **{k: rec[k] for k in ("prompt_tokens", "completion_tokens", "llm_calls", "latency_s")},
            }
            done[(path, rec["qid"])] = score
            with SCORES.open("a", encoding="utf-8") as f:
                f.write(json.dumps(score) + "\n")
            print(f"[{path}] {i + 1}/{len(records)} {rec['qid']} "
                  f"faith={score['faithfulness']} rel={score['answer_relevancy']} hit={score['hit']}")

    summarize([s for s in done.values() if s["file"] in files])


def summarize(scores):
    metrics = ["hit", "ndcg", "faithfulness", "answer_relevancy", "token_f1",
               "prompt_tokens", "completion_tokens", "llm_calls", "latency_s"]
    rows = []
    for f in dict.fromkeys(s["file"] for s in scores):
        group = [s for s in scores if s["file"] == f]
        row = {"file": f, "n": len(group)}
        for m in metrics:
            vals = [s[m] for s in group if s[m] is not None]
            row[m] = round(sum(vals) / len(vals), 3) if vals else None
        rows.append(row)
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "n", *metrics])
        w.writeheader()
        w.writerows(rows)
    print("\n| " + " | ".join(["file", "n", *metrics]) + " |")
    print("|" + "---|" * (len(metrics) + 2))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in ["file", "n", *metrics]) + " |")


if __name__ == "__main__":
    assert strip_references("Ans.\n\n### References\n\n- [1] a.txt") == "Ans."
    assert token_f1("The cat sat", ["cat sat down"]) == 0.8
    assert ndcg(["a", "b"], "a", 1) == 1.0 and ndcg(["b", "a"], "a", 1) < 1 and ndcg(["b"], "a", 3) == 0
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    asyncio.run(main(args.files, args.limit))
