#!/usr/bin/env python3

import os

import mode
from dotenv import load_dotenv
from langchain.globals import get_verbose, set_verbose
from utils import parse_args, path

# Load .env file, located in DOTENV_PATH env variable
load_dotenv(path(os.environ.get("DOTENV_PATH")))

# Turn off Langchain verbose mode
set_verbose(False)
is_verbose = get_verbose()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "eval":
        mode.eval(args)
    else:  # "ui"
        mode.ui(args)
