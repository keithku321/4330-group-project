"""Build a LightRAG instance on local Ollama, with a token tracker so cost can be measured."""
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import TokenTracker

from config import CHUNK_OVERLAP, CHUNK_SIZE, LLM_MODEL, LLM_OPTIONS, OLLAMA_HOST, RESULTS

TRACKER = TokenTracker()


async def llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await ollama_model_complete(prompt, system_prompt=system_prompt, history_messages=history_messages,
                                       token_tracker=TRACKER, **kwargs)


async def make_rag(working_dir=RESULTS / "lightrag_index", chunk_size=CHUNK_SIZE):
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host": OLLAMA_HOST, "options": LLM_OPTIONS, "timeout": 900},
        llm_model_max_async=2,          # one local GPU: more parallel calls only queue up
        embedding_func=ollama_embed,    # bge-m3, 1024-dim
        chunk_token_size=chunk_size,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        enable_llm_cache=False,         # query cost must be measured fresh every run
        enable_llm_cache_for_entity_extract=True,  # lets an interrupted index resume cheaply
    )
    await rag.initialize_storages()
    return rag
