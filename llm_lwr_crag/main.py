#!/usr/bin/env python3

from itertools import zip_longest

import pandas as pd
import pipeline as pl
from box import Box  # type: ignore
from handlers.auto import AbstractDB, AbstractLLM
from langchain.globals import get_verbose, set_verbose
from utils import logger, parse_args

# Turn off Langchain verbose mode
set_verbose(False)
is_verbose = get_verbose()


def eval(
    eval_df: pd.DataFrame,
    ret_db_vec: AbstractDB,
    ret_db_bm25: AbstractDB,
    ret_rerank: AbstractLLM,
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
    eval_df = pl.make_repo_and_eval(args)
    docs, chunks = pl.load_docs_and_chunk(args)

    ret_db_vec, ret_db_bm25, ret_rerank = pl.setup_retrieval(args, docs, chunks)
    gen_llm = pl.setup_generation(args)

    # Evaluate the dataset
    avg_recall = eval(eval_df, ret_db_vec, ret_db_bm25, ret_rerank, gen_llm)
    logger.info(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
