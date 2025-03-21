from typing import List

import progressbar
from box import Box
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(documents: List[Document], text_chunker) -> List[dict]:
    """
    Split documents into chunks.

    Args:
        documents (List[documents]): List of documents to be split into chunks.
        text_chunker: Chunker to be used for single document splitting.

    Returns:
        all_chunks (List[dict]): Combined list of all chunks, from all provided
        documents. Each chunk is represented as a dictionary, containing the text
        of the split, and the source (file it is extracted from).
    """
    all_chunks = []
    with progressbar.ProgressBar(
        widgets=[
            "Chunking documents: ",
            "[",
            progressbar.Percentage(),
            "] ",
            progressbar.Bar(),
            " ",
            progressbar.ETA(),
        ],
        max_value=len(documents),
    ) as bar:
        for i, doc in enumerate(documents):
            splits = text_chunker.split_text(doc.page_content)
            for split in splits:
                all_chunks.append({"text": split, "source": doc.metadata["path"]})

            bar.update(i + 1)

    return all_chunks


def make_text_chunker(type: str, args: Box):
    """
    Find the appropriate chunking method and return the fully initialized chunker.

    Args:
        type (str): Type of the text splitting method to use.
        args (Box): List of provided arguments, flexible to use for future
        seamless integration. Only relevant parts are extracted for specific
        text splitters.
    """
    text_chunker = None
    if type == "RecursiveCharacterTextSplitter":
        text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=args.retriever.chunking.chunk_size,
            chunk_overlap=args.retriever.chunking.chunk_overlap,
        )

    if text_chunker is None:
        raise ValueError(f"Invalid text chunker type: f{type}")

    return text_chunker
