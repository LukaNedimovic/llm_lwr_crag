import os
from pathlib import Path

__all__ = ["path"]


def path(path_as_str: str) -> Path:
    """
    Expand environment variables within the path.

    Args:
        path (str): Path to modify.

    Returns:
        path (Path): Expanded path as an pathlib.Path object.
    """
    path_as_str = os.path.expandvars(path_as_str)
    return Path(path_as_str)
