#!/usr/bin/env python3

from utils import download_repo, parse_args

if __name__ == "__main__":
    args = parse_args()

    download_repo(args.repo_url, args.repo_dir)
