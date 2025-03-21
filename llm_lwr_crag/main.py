#!/usr/bin/env python3

from box import Box  # type: ignore
from handlers.handler_factory import HandlerFactory
from utils import download_repo, parse_args, parse_eval, path


def train(args: Box):
    # Download GitHub repository and parse the evaluation data
    download_repo(args.repo_url, path(args.repo_dir))
    eval_df = parse_eval(path(args.eval_path))

    # Set up the retrieval process
    ret_db = HandlerFactory.get_db_handler(
        args.retriever.db.type,
        **(args.to_dict()),
    )
    ret_llm = HandlerFactory.get_llm_handler(
        args.retriever.llm.type,
        **(args.to_dict()),
    )
    print(eval_df)
    print(ret_db)
    print(ret_llm)


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
