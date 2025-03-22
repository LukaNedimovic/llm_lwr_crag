import os
from pathlib import Path
from typing import List, Union

import progressbar
from langchain.schema import Document
from unstructured.partition.auto import partition
from utils.logging import logger
from utils.path import path


def extract_text(file_path: Path) -> Union[str, None]:
    """
    Extracts text content from a given file using automated file type detection
    and extraction.

    Args:
        file_path (Path): Path to the file to extract text from.

    Returns:
        str: Extracted text content, or None if the file is unsupported or unreadable.
    """
    try:
        elements = partition(str(file_path))
        return "\n".join([str(el) for el in elements])
    except Exception as e:
        logger.info(f"Error extracting text from {file_path}: {e}")
        return None


def load_documents(repo_dir: Path, extensions: List[str]) -> List[Document]:
    """
    Scans the repository and loads documents with extensible extensions.

    Args:
        repo_dir (Path): Path to directory to load documents from.
        extensions (List[str]): List of extensible extensions. If a file is ending
        with any of these extensions, it should be tried to load the data from it.

    Returns:
        documents (List[Document]): List of documents with all associated metadata
        and content.
    """
    documents = []
    total_files = sum([len(files) for _, _, files in os.walk(repo_dir)])

    # Set up the progress bar
    with progressbar.ProgressBar(
        widgets=[
            "Loading documents: ",
            "[",
            progressbar.Percentage(),
            "] ",
            progressbar.Bar(),
            " ",
            progressbar.ETA(),
        ],
        max_value=total_files,
    ) as bar:
        # Walk through the repo directory and load documents
        file_count = 0
        for root, _, files in os.walk(repo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext in extensions:
                    rel_path = os.path.relpath(file_path, repo_dir)
                    text = extract_text(path(file_path))
                    if text:
                        documents.append(
                            Document(page_content=text, metadata={"path": rel_path})
                        )

                file_count += 1
                bar.update(file_count)

    return documents
