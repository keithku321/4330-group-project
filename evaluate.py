"""Score experiment answers.

    python -X utf8 evaluate.py results/exp1_naive.jsonl results/exp1_lightrag.jsonl [--limit N]

Per answer (cached in <judge dir>/scores.jsonl, so reruns only add missing answers and judgings):
  retrieval  hit rate, nDCG        doc-level: a retrieved chunk is relevant if it comes from the question's source doc
  generation faithfulness, answer relevancy   RAGAS 0.4, judge from judge_setup.py (DeepSeek via DSH, or Ollama) + bge-m3,
                                              each the mean of LIGHTRAG_JUDGE_REPEATS judgings (default 3)
             token F1 vs gold answer          SQuAD-style, cheap sanity check
  cost       prompt/completion tokens, LLM calls, latency (copied from the run)
Prints a per-system summary and writes <judge dir>/summary.csv. The judge dir is config.JUDGE_DIR: results/ itself
for the Ollama judge, results/judge_<model>/ for a DeepSeek judge, so the two judges' caches never mix.
"""
import argparse
import asyncio
import collections
import csv
import json
import math
import re
import string

from config import JUDGE_DIR, LLM_MODEL, RESULTS
from judge_setup import judge_answer, judging_complete, make_judges, save_score

SCORES = JUDGE_DIR / "scores.jsonl"


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


# Returns the model that wrote an answer record (its rag_model field, set when the answer is generated). Answers
# written before records named their model fall back to the configured LLM, the same assumption run_info.json makes,
# so score such answers with the models that produced them.
def rag_model(rec):
    return rec.get("rag_model") or LLM_MODEL


# The two model columns of a summary row: which model answered and which judged the scores in the group, joined with
# "; " if a group ever mixes several, so a mixed group is visible instead of silently averaged. "unrecorded" marks a
# score cached before score records named their models.
def model_columns(group):
    return {column: "; ".join(sorted({s.get(column) or "unrecorded" for s in group}))
            for column in ("rag_model", "judge_model")}


async def main(files, limit):
    chunks = json.loads((RESULTS / "naive_index" / "chunks.json").read_text(encoding="utf-8"))
    n_chunks = collections.Counter(c["doc_id"] for c in chunks)
    faith, relevancy = make_judges()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    done = {}
    if SCORES.exists():
        for line in SCORES.open(encoding="utf-8"):
            s = json.loads(line)
            done[(s["file"], s["qid"])] = s

    for path in files:
        records = [json.loads(l) for l in open(path, encoding="utf-8")][:limit]
        for i, rec in enumerate(records):
            # An answer is skipped only when it already has every judging; otherwise the missing ones are added.
            previous = done.get((path, rec["qid"]))
            if judging_complete(previous):
                continue
            rec["answer"] = strip_references(rec["answer"])
            score = {
                "file": path, "qid": rec["qid"], "system": rec["system"], "rag_model": rag_model(rec),
                "hit": float(rec["doc_id"] in rec["retrieved_doc_ids"]),
                "ndcg": ndcg(rec["retrieved_doc_ids"], rec["doc_id"], n_chunks[rec["doc_id"]]),
                "token_f1": token_f1(rec["answer"], rec["answers"]),
                **await judge_answer(rec["question"], rec["answer"], rec["contexts"], faith, relevancy,
                                     f"{rec['system']} {rec['qid']}", previous),
                **{k: rec[k] for k in ("prompt_tokens", "completion_tokens", "llm_calls", "latency_s")},
            }
            done[(path, rec["qid"])] = score
            save_score(SCORES, score, ("file", "qid"))
            print(f"[{path}] {i + 1}/{len(records)} {rec['qid']} faith={score['faithfulness']} "
                  f"(sd {score['faithfulness_sd']}) rel={score['answer_relevancy']} hit={score['hit']}")

    summarize([s for s in done.values() if s["file"] in files])


def summarize(scores):
    metrics = ["hit", "ndcg", "faithfulness", "answer_relevancy", "token_f1",
               "prompt_tokens", "completion_tokens", "llm_calls", "latency_s"]
    rows = []
    for f in dict.fromkeys(s["file"] for s in scores):
        group = [s for s in scores if s["file"] == f]
        row = {"file": f, "n": len(group), **model_columns(group)}
        for m in metrics:
            vals = [s[m] for s in group if s[m] is not None]
            row[m] = round(sum(vals) / len(vals), 3) if vals else None
        rows.append(row)
    columns = ["file", "n", "rag_model", "judge_model", *metrics]
    with (JUDGE_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print("\n| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in columns) + " |")


if __name__ == "__main__":
    assert strip_references("Ans.\n\n### References\n\n- [1] a.txt") == "Ans."
    assert token_f1("The cat sat", ["cat sat down"]) == 0.8
    assert ndcg(["a", "b"], "a", 1) == 1.0 and ndcg(["b", "a"], "a", 1) < 1 and ndcg(["b"], "a", 3) == 0
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    asyncio.run(main(args.files, args.limit))
