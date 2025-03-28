import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from langchain.schema import Document


def extract_text(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def process_file(file_path: Path, repo_dir: Path, extensions) -> Document:
    ext = file_path.suffix
    if ext == "":
        ext = file_path.name

    if ext not in extensions:
        print("Skipping file:", file_path, "because of extension:", file_path.suffix)
        return None

    text = extract_text(file_path)
    if text:
        return Document(
            page_content=text,
            metadata={
                "rel_path": os.path.relpath(file_path, repo_dir),
                "abs_path": str(file_path),
                "ext": file_path.suffix,
            },
        )
    return None


def load_documents(repo_dir: Path, extensions: List[str]) -> List[Document]:
    documents = []
    file_paths = [
        Path(root) / file for root, _, files in os.walk(repo_dir) for file in files
    ]

    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, fp, repo_dir, extensions): fp
            for fp in file_paths
        }

        for future in as_completed(future_to_file):
            doc = future.result()
            if doc:
                documents.append(doc)

    return documents
