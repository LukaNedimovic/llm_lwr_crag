import os
from pathlib import Path

__all__ = ["path"]


def path(path_as_str: str) -> Path:
    path_as_str = os.path.expandvars(path_as_str)
    return Path(path_as_str)
