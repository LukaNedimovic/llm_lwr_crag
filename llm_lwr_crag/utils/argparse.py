import argparse

from .path import path

__all__ = ["parse_args"]


def parse_args() -> argparse.Namespace:
    """
    Parse cmdline arguments.

    Returns:
        args (argparse.Namespace): Parsed cmdline arguments
    """
    parser = argparse.ArgumentParser(description="A simple command-line tool.")

    # Repository cloning
    parser.add_argument(
        "--repo_url", type=str, help="URL to GitHub codebase to build index from."
    )
    parser.add_argument(
        "--repo_dir", type=path, help="Local path to store the GitHub repository."
    )

    # Evaluation
    parser.add_argument("--eval_path", type=path, help="Path to evaluation data.")

    args = parser.parse_args()

    return args
