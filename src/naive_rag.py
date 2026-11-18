"""Naive RAG baseline: fixed-size chunks -> bge-m3 vectors -> cosine top-k -> one LLM call."""
import json
import time

import numpy as np
import ollama
import tiktoken

from config import (CHUNK_OVERLAP, CHUNK_SIZE, CORPUS, EMBED_MODEL, LLM_MODEL, LLM_OPTIONS,
                    NAIVE_TOP_K, OLLAMA_HOST, RESULTS, THINK)

ENC = tiktoken.get_encoding("cl100k_base")
INDEX = RESULTS / "naive_index"

PROMPT = """Answer the question using only the context below. If the context does not contain the answer, say so.
Reply in a single paragraph.

---Context---
{context}

---Question---
{question}"""


def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    toks = ENC.encode(text)
    return [ENC.decode(toks[i:i + size]) for i in range(0, max(len(toks) - overlap, 1), size - overlap)]


def top_k(query_vec, matrix, k):
    """Indices of the k rows of `matrix` most cosine-similar to `query_vec` (rows are pre-normalised)."""
    scores = matrix @ (query_vec / np.linalg.norm(query_vec))
    return np.argsort(-scores)[:k], scores


class NaiveRAG:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)
        if (INDEX / "vectors.npy").exists():
            self.vectors = np.load(INDEX / "vectors.npy")
            self.chunks = json.loads((INDEX / "chunks.json").read_text(encoding="utf-8"))
        else:
            self.build()

    def embed(self, texts):
        vecs = np.array(self.client.embed(model=EMBED_MODEL, input=texts)["embeddings"], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def build(self):
        self.chunks = [{"doc_id": p.stem, "text": c}
                       for p in sorted(CORPUS.glob("*.txt")) for c in chunk(p.read_text(encoding="utf-8"))]
        self.vectors = np.concatenate([self.embed([c["text"] for c in self.chunks[i:i + 16]])
                                       for i in range(0, len(self.chunks), 16)])
        INDEX.mkdir(parents=True, exist_ok=True)
        np.save(INDEX / "vectors.npy", self.vectors)
        (INDEX / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def query(self, question):
        t0 = time.perf_counter()
        q = self.client.embed(model=EMBED_MODEL, input=[question])
        idx, _ = top_k(np.array(q["embeddings"][0], dtype=np.float32), self.vectors, NAIVE_TOP_K)
        retrieved = [self.chunks[i] for i in idx]
        context = "\n\n".join(f"[{n + 1}] {c['text']}" for n, c in enumerate(retrieved))
        r = self.client.chat(model=LLM_MODEL, options=LLM_OPTIONS, think=THINK,
                             messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}])
        return {
            "answer": r["message"]["content"],
            "retrieved_doc_ids": [c["doc_id"] for c in retrieved],
            "contexts": [c["text"] for c in retrieved],
            "prompt_tokens": r.get("prompt_eval_count", 0),
            "completion_tokens": r.get("eval_count", 0),
            "llm_calls": 1,
            "latency_s": round(time.perf_counter() - t0, 2),
        }


if __name__ == "__main__":
    # self-check for the two pure helpers
    m = np.eye(3, dtype=np.float32)
    assert list(top_k(np.array([0.1, 0.9, 0.2], dtype=np.float32), m, 2)[0]) == [1, 2]
    pieces = chunk("word " * 3000)
    assert len(pieces) == 3 and all(len(ENC.encode(p)) <= CHUNK_SIZE for p in pieces)
    print("naive_rag self-check ok")
