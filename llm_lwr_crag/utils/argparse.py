import argparse
import os
from pathlib import Path
from typing import Union

import yaml
from box import Box
from pydantic import ValidationError

from .config_validator import ConfigValidator
from .path import path

__all__ = ["parse_args"]


def is_yaml_file(file_path: Path) -> bool:
    """
    Check whether file at the provided path is a YAML / YML file.
    """
    if not os.path.isfile(file_path):
        print(file_path)
        return False
    _, extension = os.path.splitext(file_path)
    if extension.lower() in [".yaml", ".yml"]:
        return True
    return False


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
