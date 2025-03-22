from typing import Any, List

from chromadb import PersistentClient
from chromadb.config import Settings
from utils.logging import logger
from utils.path import path

from .abstract_db_handler import AbstractDBHandler


class ChromaDBHandler(AbstractDBHandler):
    def __init__(self, args):
        self.client = PersistentClient(
            path=str(path(args.chromadb_path)),
            settings=Settings(allow_reset=True),
        )

        self.collection_name = args.collection_name
        if self.collection_name in self.client.list_collections():
            logger.info(
                f"Collection '{self.collection_name}' already exists. Dropping it..."
            )
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.create_collection(name=self.collection_name)

    def store_embeddings(
        self,
        chunks: List[str],
        embeddings: List[Any],
        metadata: List[dict],
        ids: List[str],
    ) -> None:
        """
        Store embeddings in the Chroma database.
        """
        logger.info(f"Adding embeddings into the {self.collection_name} (ChromeDB)...")
        self.collection.add(
            documents=chunks, embeddings=embeddings, metadatas=metadata, ids=ids
        )
        logger.info("Sucessfully added embeddings into the database!")

    def query(self, query_embedding: Any, top_k: int = 10) -> List[str]:
        """
        Query the Chroma database for files.
        """
        result = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        result_metadatas = result["metadatas"][0]
        sources = [metadata["source"] for metadata in result_metadatas]
        return sources

    def delete_embeddings(self, ids: List[str]) -> None:
        """
        Delete embeddings from the Chroma database.
        """
        self.collection.delete(ids=ids)
