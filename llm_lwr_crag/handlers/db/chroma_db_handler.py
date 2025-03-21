from typing import Any, List

import chromadb

from .abstract_db_handler import AbstractDBHandler


class ChromaDBHandler(AbstractDBHandler):
    def __init__(self, **kwargs):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=kwargs.get("collection_name", "default_collection")
        )

    def store_embeddings(
        self,
        documents: List[str],
        embeddings: List[Any],
        metadata: List[dict],
        ids: List[str],
    ) -> None:
        """
        Store embeddings in the Chroma database.
        """
        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids
        )

    def query(self, query_embedding: Any, n_results: int = 10) -> List[str]:
        """
        Query the Chroma database with the given embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )
        return results["documents"]

    def delete_embeddings(self, ids: List[str]) -> None:
        """
        Delete embeddings from the Chroma database.
        """
        self.collection.delete(ids=ids)
