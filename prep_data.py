"""Build the Exp corpus: a ~100k-token subset of UltraDomain mix.jsonl plus all its questions."""
import hashlib
import json

import tiktoken
from huggingface_hub import hf_hub_download

from config import CORPUS, DATA, QUESTIONS

TARGET_TOKENS = 100_000

src = DATA / "mix.jsonl"
if not src.exists():
    import shutil
    shutil.copy(hf_hub_download("TommyChien/UltraDomain", "mix.jsonl", repo_type="dataset"), src)

rows = [json.loads(line) for line in src.open(encoding="utf-8")]
enc = tiktoken.get_encoding("cl100k_base")

docs = {}  # context text -> doc id
for r in rows:
    docs.setdefault(r["context"], "doc_" + hashlib.md5(r["context"].encode()).hexdigest()[:10])

# Shortest documents first so we keep as many documents (and questions) as the budget allows.
picked, total = set(), 0
for text in sorted(docs, key=lambda t: len(enc.encode(t))):
    n = len(enc.encode(text))
    if total + n > TARGET_TOKENS:
        break
    picked.add(text)
    total += n

CORPUS.mkdir(parents=True, exist_ok=True)
for old in CORPUS.glob("*.txt"):
    old.unlink()
for text in picked:
    (CORPUS / f"{docs[text]}.txt").write_text(text, encoding="utf-8")

with QUESTIONS.open("w", encoding="utf-8") as f:
    n_q = 0
    for i, r in enumerate(rows):
        if r["context"] in picked:
            f.write(json.dumps({"qid": f"q{i:03d}", "question": r["input"], "answers": r["answers"],
                                "doc_id": docs[r["context"]], "dataset": r["dataset"]}, ensure_ascii=False) + "\n")
            n_q += 1

print(f"{len(picked)} documents, {total} tokens, {n_q} questions -> {CORPUS}, {QUESTIONS}")
