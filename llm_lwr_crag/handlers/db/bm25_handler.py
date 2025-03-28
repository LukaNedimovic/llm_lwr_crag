from typing import List, Tuple, Union

import numpy as np
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from utils.logging import logger

from .abstract_db import AbstractDB


class BM25Handler(AbstractDB):
    def __init__(self, args):
        self.db = None

    def tokenize(self, text_or_doc: Union[str, Document]) -> List[str]:
        """
        Lowercase and split the text.
        """
        text = text_or_doc
        if isinstance(text_or_doc, Document):
            text = text_or_doc.page_content

        return text.lower().split()

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Store embeddings in the BM25 index.
        """
        if self.db is None:
            logger.info("Building the M25 index...")

            tokenized_chunks = [self.tokenize(chunk) for chunk in chunks]
            self.db = BM25Okapi(tokenized_chunks)
            self.docs = chunks  # Used for future filtering purposes

            logger.info("Sucessfully created BM25 index from given files.")

    def query(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the BM25 index for files.
        """
        tok_query = self.tokenize(query)
        bm25_scores = self.db.get_scores(tok_query)

        bm25_k_idxs = np.argsort(-bm25_scores)[:k]
        bm25_k_files = [self.docs[idx] for idx in bm25_k_idxs]

        return bm25_k_files
