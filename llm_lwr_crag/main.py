#!/usr/bin/env python3

from box import Box
from utils import download_repo, parse_args, path


def train(args: Box):
    download_repo(args.repo_url, path(args.repo_dir))


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
