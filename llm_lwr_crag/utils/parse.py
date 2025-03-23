import argparse
import os
from pathlib import Path
from typing import Union

import pandas as pd  # type: ignore
import yaml
from box import Box  # type: ignore
from config.config_validator import ConfigValidator
from pydantic import ValidationError

from .path import path

__all__ = ["parse_args"]


def is_yaml_file(file_path: Path) -> bool:
    """
    Check whether file at the provided path is a YAML / YML file.
    """
    if not os.path.isfile(file_path):
        return False
    _, extension = os.path.splitext(file_path)
    return True if extension.lower() in [".yaml", ".yml"] else False


def is_json_file(file_path: Path) -> bool:
    """
    Check whether file at the provided path is a JSON file.
    """
    if not os.path.isfile(file_path):
        return False
    _, extension = os.path.splitext(file_path)
    return True if extension.lower() == ".json" else False


def is_txt_file(file_path: Path) -> bool:
    """
    Check whether file at the provided path is a TXT file.
    """
    if not os.path.isfile(file_path):
        return False
    _, extension = os.path.splitext(file_path)
    return True if extension.lower() == ".txt" else False


def parse_config(path: Path) -> Union[None, Box]:
    """
    Parse configuration YAML file.

    Args:
        path (Path): Path to the configuration YAML file.

    Returns:
        config (None, Box): If invalid path or configuration, return None.
        Otherwise, return the wrapped configuration object.
    """
    if not is_yaml_file(path):
        return None

    with open(path, "r") as file:
        config = yaml.safe_load(file)

        try:
            config = ConfigValidator(**config)
            return Box(config.model_dump())
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None


def parse_txt(path: Path) -> Union[None, str]:
    """
    Parse a standard TXT file into a string.

    Args:
        path (Path): Path to TXT file to parse.

    Returns:
        Content of provided TXT file as a single string.
    """
    if not is_txt_file(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def parse_eval(path: Path) -> pd.DataFrame:
    return pd.read_json(path) if is_json_file(path) else None


def parse_args() -> Union[None, Box]:
    """
    Parse cmdline arguments.

    Returns:
        args (DotDict): Parsed cmdline arguments, wrapped inside a DotDict object
        for easier access.
    """
    parser = argparse.ArgumentParser(
        description="LLM Listwise Reranking for CodeRAG parser."
    )
    parser.add_argument("--config", type=path, help="Path to YAML configuration file.")

    args = parser.parse_args()

    config = parse_config(args.config)

    return config
