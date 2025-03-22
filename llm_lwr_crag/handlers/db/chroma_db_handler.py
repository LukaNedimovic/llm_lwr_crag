from typing import Any, List

from chromadb import Client
from chromadb.config import Settings

from .abstract_db_handler import AbstractDBHandler


class ChromaDBHandler(AbstractDBHandler):
    def __init__(self, args):
        self.client = Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=args.chromadb_path,
            )
        )
        self.collection_name = args.collection_name

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
        self.collection.add(
            documents=chunks, embeddings=embeddings, metadatas=metadata, ids=ids
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
