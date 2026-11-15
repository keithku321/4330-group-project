"""Collect the local results into share/ for the report and teammates (local twin of notebook section 10).

    python -X utf8 export.py

share/per_question.csv  one row per question x system, with the answer and every score
share/run_info.json     models, settings, corpus size, index cost
share/*_summary.csv     the per-system means printed by evaluate.py / exp2.py / exp3.py

Scores are read from the configured judge's folder (config.JUDGE_DIR), so run export.py with the same judge settings
as the scoring run. A LIGHTRAG_SUBSET run exports to results/<subset>/share/ and never overwrites the committed share/.
"""
import csv
import json
import shutil

import tiktoken
from lightrag import QueryParam

import config as C
from exp2 import DEFAULT, SETTINGS, answers_path, system
from exp3 import MODES, load_jsonl

# R holds the answers and index costs, J the scores and summaries of the configured judge (R itself for Ollama).
R, J = C.RESULTS, C.JUDGE_DIR
SHARE = C.RESULTS / "share" if C.SUBSET else C.ROOT / "share"
QUESTIONS = load_jsonl(C.QUESTIONS)
k_default = QueryParam().top_k

# (experiment, answers file, score file, system name in the score file, mode label, chunk size, top-k)
sources = [("exp1", R / "exp1_naive.jsonl", J / "scores.jsonl", "naive", "naive_rag", C.CHUNK_SIZE, C.NAIVE_TOP_K),
           ("exp1", R / "exp1_lightrag.jsonl", J / "scores.jsonl", "lightrag", "lightrag-mix", C.CHUNK_SIZE, k_default)]
sources += [("exp3", R / f"exp3_{m}.jsonl", J / "exp3_scores.jsonl", f"lightrag-{m}", f"lightrag-{m}", C.CHUNK_SIZE,
             k_default) for m in MODES]
# Exp 2's default setting is the Exp 3 mix run, already above
sources += [("exp2", answers_path(cs, k), J / "exp2_scores.jsonl", system(cs, k), "lightrag-mix", cs, k)
            for cs, k in SETTINGS if (cs, k) != DEFAULT]

q_by_id = {q["qid"]: q for q in QUESTIONS}
rows = []
for exp, answers, score_file, sys_name, mode, cs, k in sources:
    scores = {s["qid"]: s for s in load_jsonl(score_file) if s["system"] == sys_name}
    for r in load_jsonl(answers):
        s = scores.get(r["qid"])
        if s is None:  # answered but not scored yet
            continue
        rows.append({"experiment": exp, "qid": r["qid"], "question": r["question"], "system": sys_name, "mode": mode,
                     "chunk_size": cs, "top_k": k, "hit": s["hit"], "ndcg": round(s["ndcg"], 4), "answer": r["answer"],
                     "gold_answer": " | ".join(r["answers"]), "token_f1": round(s["token_f1"], 4),
                     "faithfulness": s["faithfulness"], "answer_relevancy": s["answer_relevancy"],
                     "n_entities": r.get("n_entities"), "n_relations": r.get("n_relations"),
                     "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
                     "llm_calls": s["llm_calls"], "latency_s": s["latency_s"],
                     "dataset": q_by_id[r["qid"]]["dataset"], "source_doc": r["doc_id"]})
# With no scores in the configured judge's folder (e.g. the DeepSeek judge before any rescoring), stop before touching
# share/, rather than truncating the committed per_question.csv and crashing on rows[0].
if not rows:
    raise SystemExit(f"No scored answers in {J} for judge '{C.JUDGE_NAME}'; nothing exported. Score first, or set "
                     "LIGHTRAG_JUDGE_BACKEND to the judge that produced the scores.")
SHARE.mkdir(exist_ok=True)
with open(SHARE / "per_question.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


docs = sorted(C.CORPUS.glob("*.txt"))
texts = [p.read_text(encoding="utf-8") for p in docs]
enc = tiktoken.get_encoding("cl100k_base")
# Records the models config.py actually resolved (primary, Qwen 3.5 fallback, DeepSeek judge or an environment
# override) rather than fixed names, so a re-export after a fallback run cannot claim the qwen2.5 judge. The think
# flags are None when nothing was sent (qwen2.5, or a DeepSeek judge, whose setting is judge_effort) and False when
# thinking was switched off; they describe this export's configuration, so run export.py with the same models and
# environment variables as the run that produced results/.
run_info = {
    "backend": "Ollama (local GPU)", "llm": C.LLM_MODEL, "llm_options": C.LLM_OPTIONS, "llm_think": C.THINK,
    "embedding": C.EMBED_MODEL, "embedding_dim": C.EMBED_DIM,
    "judge": f"{C.JUDGE_NAME} (RAGAS 0.4 faithfulness + answer relevancy)", "judge_backend": C.JUDGE_BACKEND,
    "judge_think": C.JUDGE_THINK, "judge_effort": C.DSH_JUDGE_EFFORT if C.JUDGE_BACKEND == "dsh" else None,
    "chunk_size": C.CHUNK_SIZE, "chunk_overlap": C.CHUNK_OVERLAP, "naive_top_k": C.NAIVE_TOP_K,
    "lightrag_top_k": k_default, "lightrag_chunk_top_k": QueryParam().chunk_top_k,
    "lightrag_max_context_tokens": C.MAX_CONTEXT_TOKENS, "reranker": False, "exp2_settings": SETTINGS,
    "documents": len(docs), "corpus_tokens_cl100k": sum(len(enc.encode(t)) for t in texts),
    "corpus_words": sum(len(t.split()) for t in texts),
    "chunks_at_1200": len(json.loads((R / "naive_index" / "chunks.json").read_text(encoding="utf-8"))),
    "questions": len(QUESTIONS),
    "index_cost_c1200_lower_bound": read_json(R / "exp1_index_cost.json"),
    "index_cost_c600": read_json(R / "exp2_index_cost_c600.json"),
    "keyword_diagnostic": read_json(R / "exp3_keyword_diagnostic.json"),
}
(SHARE / "run_info.json").write_text(json.dumps(run_info, indent=2))
# A summary this judge has not produced yet is removed, so share/ never pairs one judge's run_info.json with another
# judge's leftover summary.
for name in ["summary.csv", "exp2_summary.csv", "exp3_summary.csv"]:
    target = SHARE / ("exp1_summary.csv" if name == "summary.csv" else name)
    if (J / name).exists():
        shutil.copy(J / name, target)
    else:
        target.unlink(missing_ok=True)
print(len(rows), "rows ->", SHARE / "per_question.csv")
print(json.dumps(run_info, indent=2))
