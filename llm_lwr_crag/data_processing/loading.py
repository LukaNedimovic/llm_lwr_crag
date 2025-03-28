import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

from box import Box
from langchain.schema import Document

from .metadata import add_doc_metadata


def extract_text(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def process_file(
    file_path: Path,
    repo_dir: Path,
    extensions: List[str],
    metadata_args: Box,
) -> Union[None, Document]:
    """
    Processes files by loading their content and appending the relevant
    file metadata.

    Args:
        file_path (Path): Path to the file to process.
        repo_dir (Path): Root to the repo directory - useful for path construction.
        extensions (List[str]): List of valid extensions to use.
    Returns:
        Union[None, Document]: Processed file as a Document, if valid.
            Otherwise, will return None.
    """

    # File extension to be constructed based on the actual extension if present
    # Otherwise, use the full filename (e.g. LICENSE)
    ext = file_path.suffix
    if ext == "":
        ext = file_path.name

    # Filter out files with extensions not of interest
    if ext not in extensions:
        # print("Skipping file:", file_path, "because of extension:", file_path.suffix)
        return None

    # Extract text, and, if present, construct the file with relevant metadata
    text = extract_text(file_path)
    if text:
        doc = Document(
            page_content=text,
            metadata={
                "rel_path": os.path.relpath(file_path, repo_dir),
                "abs_path": str(file_path),
                "ext": file_path.suffix,
            },
        )
        add_doc_metadata(doc, metadata_args)
        return doc

    return None


def load_docs(
    repo_dir: Path, extensions: List[str], metadata_args: Box
) -> List[Document]:
    documents = []
    file_paths = [
        Path(root) / file for root, _, files in os.walk(repo_dir) for file in files
    ]

    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, fp, repo_dir, extensions, metadata_args): fp
            for fp in file_paths
        }

        for future in as_completed(future_to_file):
            doc = future.result()
            if doc:
                documents.append(doc)

    return documents
