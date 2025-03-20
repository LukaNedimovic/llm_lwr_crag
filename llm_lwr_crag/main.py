#!/usr/bin/env python3

from utils.argparse import parse_args
from utils.download import download_repo
from utils.logging import setup_logger

if __name__ == "__main__":
    setup_logger()
    args = parse_args()

    download_repo(args.repo_url, args.repo_dir)
