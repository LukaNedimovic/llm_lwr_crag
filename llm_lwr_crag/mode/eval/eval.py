import utils.pipeline as pl
from box import Box  # type: ignore
from rag import RAG
from utils import log_res, logger, path


def eval(args: Box) -> None:
    """
    Evaluate RAG on given dataset.
    General evaluation process:
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
    # Download repo and parse evaluation data
    # Repo is not being returned since it is downloaded to local
    # Evaluation dataset is wrapped into a `pandas.DataFrame` object
    eval_df = pl.make_repo_and_eval(args)
    docs, chunks = pl.load_docs_and_chunk(args)  # Load and chunk documents
    rag = RAG.from_args(args, docs, chunks)  # Set up RAG with given docs / chunks

    # Evaluate RAG on dataset
    # log_path is the place of the local log file
    avg_recall = rag.eval(eval_df, k=args.retriever.k)

    # Log if applicable
    log_res(
        log_path=path(args.log_path),
        log_dict={
            "exp_name": args.exp_name,
            "eval": args.retriever.get("eval", None),
            "ret_chunker": args.retriever.chunking.type,
            "chunk_size": args.retriever.chunking.chunk_size,
            "chunk_overlap": args.retriever.chunking.chunk_overlap,
            "num_docs": str(len(docs)),
            "num_chunks": str(len(chunks)),
            "metadata": args.retriever.metadata,
            "ret_vec_db": str(rag.retriever.vec_db),
            "ret_db_bm25": str(rag.retriever.bm25),
            "ret_rerank": str(rag.retriever.rerank),
            "gen_llm": str(rag.generator.llm),
            "avg_recall": avg_recall,
            "k": args.retriever.k,
        },
    )

    logger.info(f"{avg_recall * 100:.2f}")
