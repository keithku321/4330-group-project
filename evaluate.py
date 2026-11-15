"""Score experiment answers.

    python -X utf8 evaluate.py results/exp1_naive.jsonl results/exp1_lightrag.jsonl [--limit N]

Per answer (cached in <judge dir>/scores.jsonl, so reruns only score new answers):
  retrieval  hit rate, nDCG        doc-level: a retrieved chunk is relevant if it comes from the question's source doc
  generation faithfulness, answer relevancy   RAGAS 0.4, judge from judge_setup.py (DeepSeek via DSH, or Ollama) + bge-m3
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

from config import JUDGE_DIR, RESULTS
from judge_setup import JudgeUnavailableError, judge_label, make_judges

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


# Scores one answer with both RAGAS metrics. Each metric is labelled for the DSH session list before it runs. A judge
# that is unusable (JudgeUnavailableError) stops the run; any other failure loses only that one score.
async def ragas_scores(rec, faith, relevancy):
    out = {}
    for name, score in [("faithfulness", lambda: faith.ascore(user_input=rec["question"], response=rec["answer"],
                                                              retrieved_contexts=rec["contexts"] or [""])),
                        ("answer_relevancy", lambda: relevancy.ascore(user_input=rec["question"],
                                                                      response=rec["answer"]))]:
        judge_label.set(f"{rec['system']} {rec['qid']} {name}")
        try:
            out[name] = (await score()).value
        except JudgeUnavailableError:
            raise
        except Exception as e:  # a malformed judge reply loses one score, not the whole run
            print(f"  {name} failed for {rec['system']}/{rec['qid']}: {str(e)[:150]}")
            out[name] = None
    return out


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
    with (JUDGE_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
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
