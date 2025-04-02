#!/usr/bin/env python3


import pipeline as pl
from box import Box  # type: ignore
from langchain.globals import get_verbose, set_verbose
from rag import RAG
from utils import logger, parse_args

# Turn off Langchain verbose mode
set_verbose(False)
is_verbose = get_verbose()


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
    docs, chunks = pl.load_docs_and_chunk(args)  # Load and chunk documents
    rag = RAG.from_args(args, docs, chunks)  # Set up RAG with given docs / chunks

    # Evaluate RAG on dataset
    avg_recall = rag.eval(eval_df, k=args.retriever.k)
    logger.info(f"{avg_recall * 100:.2f}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
