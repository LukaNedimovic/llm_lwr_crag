#!/usr/bin/env python3

from box import Box  # type: ignore
from data_processing import chunk_documents, load_documents, make_text_chunker
from handlers.handler_factory import HandlerFactory
from utils import download_repo, gen_extensions, parse_args, parse_eval, path


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
    chunks = chunk_documents(documents, text_chunker)  # noqa: F841

    # Database and LLM used for embedding the chunks
    ret_db = HandlerFactory.get_db_handler(  # noqa: F841
        args.retriever.db.type,
        **(args.to_dict()),
    )
    ret_llm = HandlerFactory.get_llm_handler(  # noqa: F841
        args.retriever.llm.type,
        **(args.to_dict()),
    )


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
