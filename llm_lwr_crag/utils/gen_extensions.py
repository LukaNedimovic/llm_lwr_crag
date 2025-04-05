import os
from pathlib import Path
from typing import Set

import yaml

from .logging import logger


def load_extensions(extensions_path: Path) -> Set[str]:
    """
    Read extensions from already present file.

    Args:
        path (Path): Path to .txt file to read extensions from.

    Returns:
        extensions (List[str]): List of extensions stored within the file.
    """
    extensions = set()

    try:
        with open(extensions_path, "r") as ext_file:
            for line in ext_file:
                extension = line.strip()
                if extension:
                    extensions.add(extension)
        logger.info(f"Extensions have been loaded from {extensions_path}")
    except FileNotFoundError:
        logger.error(f"Error: {extensions_path} not found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return extensions


def save_extensions(extensions: Set[str], extensions_path: Path):
    """
    Save extensions to local.

    Args:
        extensions (List[str]): List of extensions to save.
        extensions_path (Path): Path to the local storage.
    """
    with open(extensions_path, "w") as ext_file:
        for ext in extensions:
            ext_file.write(f"{ext}\n")
    logger.info(f"Extensions have been saved to {extensions_path}")


def add_std_additional_extensible(extensions: Set[str]):
    """
    Add standard configuration and documentation extensions.
    Modify the `extensions` list in-place.

    Args:
        extensions (List[str]): List of extensions to expand.
    """
    additional_extensions = {
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".env",
        ".cfg",
        ".md",
        ".rst",
        # ".txt",
        ".adoc",
        ".npmrc",
        ".nvmdrc",
        ".CN",
        ".json",
        ".md",
        ".csv",
        ".xml",
        # "LICENSE",  # For files like LICENSE
    }
    extensions.update(additional_extensions)


def rem_std_nonextensbile(extensions: Set[str]):
    """
    Remove standard non-extensible extensions, such as the ones of images,
    executables, zipped files and similar.
    Modify the `extensions` list in-place.

    Args:
        extensions (List[str]): List of extensions to remove from.
    """
    non_extensions = {
        ".png",
        ".jpg",
        ".gif",
        ".exe",
        ".dll",
        ".so",
        ".zip",
        ".tar.gz",
        ".log",
        ".cmd",
        ".bat",
        ".vbs",
    }
    extensions -= non_extensions


def gen_extensions(
    languages_path: Path, extensions_path: Path = Path(""), force: bool = False
) -> Set[str]:
    """
    Dynamically generate a list of extensible extensions for RAG system,
    trying to skip manual addition / removal as much as possible.

    Args:
        languages_path (Path): Document containing languages and associated
        file extensions. By pruning this file, generate the list of meaningful
        extensions.

    Returns:
        extensions (List[str]): List of useful extensions for a RAG system.
    """
    if os.path.exists(extensions_path):
        if force:
            os.remove(extensions_path)
        else:
            extensions = load_extensions(extensions_path)
            return extensions

    with open(languages_path, "r") as file:
        languages = yaml.safe_load(file)

    extensions = set()
    for lang, data in languages.items():
        if data.get("type") == "programming" or data.get("type") == "markup":
            for ext in data.get("extensions", []):
                extensions.add(ext)

    add_std_additional_extensible(extensions)
    rem_std_nonextensbile(extensions)

    if extensions_path is not None:
        save_extensions(extensions, extensions_path)

    return extensions
