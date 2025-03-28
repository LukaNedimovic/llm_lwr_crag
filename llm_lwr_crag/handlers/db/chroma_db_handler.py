from typing import List, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma
from utils.logging import logger

from .abstract_db import AbstractDB


class ChromaDBHandler(AbstractDB):
    def __init__(self, args):
        self.collection_name = args.collection_name
        self.db = Chroma(
            collection_name=args.collection_name,
            embedding_function=args.emb_func,
        )

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Store embeddings in the Chroma database.
        """
        logger.info(f"Adding embeddings into the {self.collection_name} (ChromeDB)...")
        self.db.add_documents(chunks)
        logger.info("Sucessfully added embeddings into the database!")

    def query(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the Chroma database for files.
        This modified version returns both the file paths and their similarity scores.
        """
        ret_chunks = self.db.similarity_search(query, k=k)
        return ret_chunks
