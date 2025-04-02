#!/usr/bin/env python3


import pandas as pd
import pipeline as pl
from box import Box  # type: ignore
from langchain.globals import get_verbose, set_verbose
from rag import RAG
from utils import logger, parse_args
from utils.logging import log_tc

# Turn off Langchain verbose mode
set_verbose(False)
is_verbose = get_verbose()


def eval(
    rag: RAG,
    eval_df: pd.DataFrame,
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

    for tc_id, row in eval_df.iterrows():
        query = row["question"]
        ground_truth_fps = set(row["files"])

        # Retrieve relevant file paths and chunks
        ret_fps, ret_chunks, gen_ans = rag(query, k)

        # Calculate Recall@K for the query
        recall, ret_relevant = RAG.recall(ret_fps, ground_truth_fps)
        total_recall += recall

        # Log the test case
        log_tc(
            tc_id=tc_id,
            num_tc=len(eval_df),
            query=query,
            ground_truth_fps=ground_truth_fps,
            ret_fps=ret_fps,
            ret_relevant=ret_relevant,
            recall=recall,
            gen_ans=gen_ans,
        )

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
    # Download repo, parse evaluation data
    eval_df = pl.make_repo_and_eval(args)

    # Load documents and chunk them
    docs, chunks = pl.load_docs_and_chunk(args)

    # Setup RAG
    ret_db_vec, ret_db_bm25, ret_rerank = pl.setup_retrieval(args, docs, chunks)
    gen_llm = pl.setup_generation(args)

    # Create a RAG pipeline
    rag = RAG(ret_db_vec, ret_db_bm25, ret_rerank, gen_llm)

    # Evaluate RAG on dataset
    avg_recall = eval(rag, eval_df, k=args.retriever.k)
    logger.info(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
