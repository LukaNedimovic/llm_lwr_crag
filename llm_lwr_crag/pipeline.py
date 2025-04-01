from typing import List, Tuple

import pandas as pd
from box import Box
from data_processing import (
    chunk_docs,
    load_docs,
    make_text_chunker,
    preprocess_eval,
)
from handlers.auto import AutoDB, AutoLLM
from langchain.schema import Document
from utils import download_repo, gen_extensions, parse_eval, path


def make_repo_and_eval(args: Box) -> pd.DataFrame:
    # Download GitHub repository and parse the evaluation data
    download_repo(args.repo_url, path(args.repo_dir), force_download=False)
    eval_df = parse_eval(path(args.eval_path))  # noqa: F841
    preprocess_eval(eval_df, args.retriever.eval)

    return eval_df


def load_docs_and_chunk(args: Box) -> Tuple[List[Document], List[Document]]:
    extensions = gen_extensions(
        path(args.languages_path),
        extensions_path=path(args.extensions_path),
        force=True,
    )
    docs = load_docs(path(args.repo_dir), extensions, args.retriever.metadata)
    text_chunker = make_text_chunker(args.retriever.chunking)
    chunks = chunk_docs(docs, text_chunker)

    return docs, chunks


def setup_generation(args: Box) -> AutoLLM:
    gen_llm = None

    # Answer generation using LLM
    if args.generator:
        gen_llm = AutoLLM.from_args(args.generator)

    return gen_llm


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
