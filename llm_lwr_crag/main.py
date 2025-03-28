#!/usr/bin/env python3

from itertools import zip_longest

import pandas as pd
from box import Box  # type: ignore
from data_processing import (
    chunk_documents,
    load_documents,
    make_text_chunker,
)
from handlers.auto import AbstractDB, AutoDB, AutoLLM
from utils import (
    download_repo,
    gen_extensions,
    logger,
    parse_args,
    parse_eval,
    path,
)


def eval(
    ret_db: AbstractDB,
    eval_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """
    Evaluate the system with Recall@K metric.

    Args:
        ret_db (AbstractDBHandler): Database to query.
        eval_df (pd.DataFrame): Evaluation data.

    Returns:
        avg_recall (float): Average Recall@K (defaults to Recall@10) over the
        evaluation dataset.
    """
    total_recall = 0.0

    for _, row in eval_df.iterrows():
        # Query the database to get top-K relevant files
        retrieved_files_all = ret_db.query(row["question"], k=4 * k)
        retrieved_files = set()
        for rf in retrieved_files_all:
            retrieved_files.add(rf.metadata["rel_path"])

        # Calculate Recall@K for the question
        ground_truth_files = set(row["files"])
        relevant_retrieved = ground_truth_files.intersection(retrieved_files)

        recall = (
            len(relevant_retrieved) / len(ground_truth_files)
            if ground_truth_files
            else 0.0
        )

        total_recall += recall

        for gnd, ret in zip_longest(ground_truth_files, retrieved_files, fillvalue=""):
            logger.info(f"{str(gnd):<80} {str(ret)}")
        logger.info(f"Common:  len(relevant_retrieved) {recall * 100:.2f}")

    # Average Recall@K across all questions
    avg_recall = total_recall / len(eval_df)
    return avg_recall


def train(args: Box):
    # Download GitHub repository and parse the evaluation data
    download_repo(args.repo_url, path(args.repo_dir), force_download=False)
    eval_df = parse_eval(path(args.eval_path))  # noqa: F841

    # Set up retrieval pipeline
    # Document loading and chunking
    extensions = gen_extensions(
        path(args.languages_path),
        extensions_path=path(args.extensions_path),
        force=True,
    )
    documents = load_documents(path(args.repo_dir), extensions)
    text_chunker = make_text_chunker(args.retriever.chunking)
    chunks = chunk_documents(documents, text_chunker)

    # LLM used for embedding the chunks and database initialization
    ret_emb = AutoLLM.from_args(args.retriever.llm)
    args.retriever.db["emb_llm"] = ret_emb
    ret_db = AutoDB.from_args(args.retriever.db)
    ret_db.add_documents(chunks)

    # Evaluate the dataset
    avg_recall = eval(ret_db, eval_df)
    logger.info(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
