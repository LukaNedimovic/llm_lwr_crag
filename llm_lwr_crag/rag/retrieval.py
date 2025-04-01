from typing import List, Tuple

from box import Box
from handlers.auto import AutoDB, AutoLLM
from langchain.schema import Document


def setup_retrieval(
    args: Box, docs: List[Document], chunks: List[Document]
) -> Tuple[AutoDB, AutoDB, AutoLLM]:
    # LLM used for embedding the chunks
    ret_emb_llm = AutoLLM.from_args(args.retriever.llm)

    # Add the model to the kwargs and create the database with the LLM
    # as embedding function
    args.retriever.db["emb_func"] = ret_emb_llm
    ret_db_vec = AutoDB.from_args(args.retriever.db)
    ret_db_vec.add_documents(chunks)

    # BM25 setup, for hybrid search
    ret_db_bm25 = None
    if args.retriever.bm25:
        ret_db_bm25 = AutoDB.from_args(Box({"provider": "bm25"}))
        if args.retriever.bm25 == "docs":
            print("Adding docs into BM25...")
            ret_db_bm25.add_documents(docs)
        else:
            print("Adding chunks into BM25...")
            ret_db_bm25.add_documents(chunks)

    # Reranking LLM setup
    ret_rerank = None
    if args.retriever.rerank:
        ret_rerank = AutoLLM.from_args(args.retriever.rerank)

    return ret_db_vec, ret_db_bm25, ret_rerank
