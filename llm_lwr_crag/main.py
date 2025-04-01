#!/usr/bin/env python3

from itertools import zip_longest

import pandas as pd
from box import Box  # type: ignore
from data_processing import (
    chunk_docs,
    load_docs,
    make_text_chunker,
    preprocess_eval,
)
from handlers.auto import AbstractDB, AbstractLLM, AutoDB, AutoLLM

# Turn off Langchain verbose mode
from langchain.globals import get_verbose, set_verbose
from utils import (
    download_repo,
    gen_extensions,
    logger,
    parse_args,
    parse_eval,
    path,
)

set_verbose(False)
is_verbose = get_verbose()


def eval(
    eval_df: pd.DataFrame,
    ret_db_vec: AbstractDB,
    ret_rerank: AbstractLLM,
    ret_db_bm25: AbstractDB,
    gen_llm: AbstractLLM,
    k: int = 10,
) -> float:
    """
    Evaluate the system with Recall@K metric.

    Args:
        ret_db_vec (AbstractDBHandler): Database to query.
        eval_df (pd.DataFrame): Evaluation data.
        k (int = 10): Top-K files to be retrieved

    Returns:
        avg_recall (float): Average Recall@K (defaults to Recall@10) over the
        evaluation dataset.
    """
    total_recall = 0.0

    for test_id, row in eval_df.iterrows():
        query = row["question"]

        # Query the database to get top-K relevant files
        # Corrective factor of 4 used as an example - can be changed
        # as it is a hyperparameter
        ret_chunks = ret_db_vec.query(query, k=4 * k)
        ret_fps = None

        # Apply BM25, if applicable
        # Hybrid search with BM25 is done based on file paths,
        # therefore file paths must be filtered out first,
        # and then the chunks
        if ret_db_bm25:
            ret_chunks_bm25 = ret_db_bm25.query(query, k=k // 2)
            ret_fps_bm25, ret_chunks_bm25 = AbstractDB.filter_by_fp(
                ret_chunks_bm25, top_k=k // 2
            )

            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)
            ret_fps = AbstractDB.rbf(ret_fps, ret_fps_bm25)
            ret_chunks = AbstractDB.fetch_by_fp(ret_fps, ret_chunks + ret_chunks_bm25)

        # Apply reranking, if applicable
        # Reranking is done based on the file content
        # Filter out the files after reranking them
        if ret_rerank:
            ret_chunks = ret_rerank.rerank(query, ret_chunks)
            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)

        # In case over pure vector search, make sure to filter out file paths
        if ret_fps is None:
            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)
        # Calculate Recall@K for the question
        ground_truth_fps = set(row["files"])
        relevant_retrieved = ground_truth_fps.intersection(ret_fps)

        # Calculate Recall@K and add it to the total_recall for future calculation
        recall = (
            len(relevant_retrieved) / len(ground_truth_fps) if ground_truth_fps else 0.0
        )
        total_recall += recall

        logger.info(f"Test: {test_id + 1} / {len(eval_df)}")
        logger.info(f"Query: {query}")
        # Compare the ground truth values and the retrieved files
        for gnd, ret in zip_longest(ground_truth_fps, ret_fps, fillvalue=""):
            logger.info(f"{str(gnd):<80} {str(ret)}")
        logger.info(
            (
                f"Common: {len(relevant_retrieved)} / {len(ground_truth_fps)} "
                f"{recall * 100:.2f}"
            )
        )

        if gen_llm:
            logger.info(f"Answer: {gen_llm.generate(query, ret_chunks)}")

    # Average Recall@K across all questions
    avg_recall = total_recall / len(eval_df)
    return avg_recall


def train(args: Box) -> None:
    """
    Perform training and evaluation.
    General training process:
        (1) Download the GitHub directory
        (2) Load the evaluation dataset
        (3) Generate extensions, or load them if already present
        (4) Load documents, alongside their designated metadata
        (5) Chunk the documents
        (6) Set up retrieval embedding function (generally, an embedding LLM)
        (7) Set up retrieval database, using the aforementioned emb. function
        (8) Add documents to the database
        (9) Evaluate

    Args:
        args (Box): Parsed YML file configuration arguments.
            Will be used in several setup steps.

    Returns:
        None
    """
    # Download GitHub repository and parse the evaluation data
    download_repo(args.repo_url, path(args.repo_dir), force_download=False)
    eval_df = parse_eval(path(args.eval_path))  # noqa: F841
    preprocess_eval(eval_df, args.retriever.eval)

    # Set up retrieval pipeline
    # Document loading and chunking
    extensions = gen_extensions(
        path(args.languages_path),
        extensions_path=path(args.extensions_path),
        force=True,
    )
    docs = load_docs(path(args.repo_dir), extensions, args.retriever.metadata)
    text_chunker = make_text_chunker(args.retriever.chunking)
    chunks = chunk_docs(docs, text_chunker)

    # LLM used for embedding the chunks
    ret_emb_llm = AutoLLM.from_args(args.retriever.llm)

    # Add the model to the kwargs and create the database with the LLM
    # as embedding function
    args.retriever.db["emb_func"] = ret_emb_llm
    ret_db_vec = AutoDB.from_args(args.retriever.db)
    ret_db_vec.add_documents(chunks)

    # Reranking LLM setup
    ret_rerank = None
    if args.retriever.rerank:
        ret_rerank = AutoLLM.from_args(args.retriever.rerank)

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

    # Answer generation using LLM
    if args.generator:
        gen_llm = AutoLLM.from_args(args.generator)

    # Evaluate the dataset
    avg_recall = eval(eval_df, ret_db_vec, ret_rerank, ret_db_bm25, gen_llm)
    logger.info(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
