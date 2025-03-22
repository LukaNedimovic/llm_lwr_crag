#!/usr/bin/env python3

from itertools import zip_longest

import pandas as pd
from box import Box  # type: ignore
from data_processing import chunk_documents, load_documents, make_text_chunker
from handlers.handler_factory import (AbstractDBHandler, AbstractLLMHandler,
                                      HandlerFactory)
from utils import download_repo, gen_extensions, parse_args, parse_eval, path


def eval(
    ret_llm: AbstractLLMHandler,
    ret_db: AbstractDBHandler,
    eval_df: pd.DataFrame,
    k: int = 10,
):
    """
    Evaluate the system with Recall@K metric.

    Args:
        ret_db (AbstractDBHandler): Database to query.
        eval_df (pd.DataFrame): Evaluation data.
    """
    total_recall = 0.0

    for _, row in eval_df.iterrows():
        # Query the database to get top-K relevant files
        question = ret_llm.embed_text(row["question"])
        retrieved_files = ret_db.query(question, top_k=k)
        retrieved_files = set(retrieved_files)

        # Calculate Recall@K for this question
        ground_truth_files = set(row["files"])
        relevant_retrieved = ground_truth_files.intersection(retrieved_files)
        recall = (
            len(relevant_retrieved) / len(ground_truth_files)
            if ground_truth_files
            else 0.0
        )

        total_recall += recall

        for gnd, ret in zip_longest(ground_truth_files, retrieved_files, fillvalue=""):
            print(f"{str(gnd):<80} {str(ret)}")
        print("Common: ", len(relevant_retrieved))

    # Average Recall@K across all questions
    avg_recall = total_recall / len(eval_df)
    return avg_recall


def train(args: Box):
    # Download GitHub repository and parse the evaluation data
    download_repo(args.repo_url, path(args.repo_dir))
    eval_df = parse_eval(path(args.eval_path))  # noqa: F841

    # Set up retrieval pipeline
    # Document loading and chunking
    extensions = gen_extensions(
        path(args.languages_path),
        extensions_path=path(args.extensions_path),
    )
    documents = load_documents(path(args.repo_dir), extensions)
    text_chunker = make_text_chunker(args.retriever.chunking.type, args)
    chunks = chunk_documents(documents, text_chunker)

    # LLM used for embedding the chunks and Database initialization
    ret_llm = HandlerFactory.get_llm_handler(args.retriever.llm)
    ret_db = HandlerFactory.get_db_handler(args.retriever.db)

    chunks_embedding_data = ret_llm.embed_chunks(chunks)
    chunks_text = [chunk["text"] for chunk in chunks]
    ret_db.store_embeddings(
        chunks_text,
        chunks_embedding_data["embeddings"],
        chunks_embedding_data["metadata"],
        chunks_embedding_data["ids"],
    )

    avg_recall = eval(ret_llm, ret_db, eval_df)
    print(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
