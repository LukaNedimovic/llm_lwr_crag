from typing import Any, List

from chromadb import PersistentClient
from chromadb.config import Settings
from utils.logging import logger
from utils.path import path

from .abstract_db import AbstractDB


class ChromaDBHandler(AbstractDB):
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

    def store_embeddings(self, chunks_data: dict) -> None:
        """
        Store embeddings in the Chroma database.
        """
        logger.info(f"Adding embeddings into the {self.collection_name} (ChromeDB)...")
        self.collection.add(
            documents=chunks_data["texts"],
            embeddings=chunks_data["embeddings"],
            metadatas=chunks_data["metadatas"],
            ids=chunks_data["ids"],
        )
        logger.info("Sucessfully added embeddings into the database!")

    def query(self, query_embedding: Any, top_k: int = 10) -> List[str]:
        """
        Query the Chroma database for files.
        """
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["metadatas", "distances"],  # Include distances in the result
        )
        result_metadatas = result["metadatas"][0]
        result_distances = result["distances"][0]
        metadata_distance_pairs = list(zip(result_metadatas, result_distances))
        metadata_distance_pairs.sort(key=lambda x: x[1])
        sorted_sources = [metadata["source"] for metadata, _ in metadata_distance_pairs]

        return sorted_sources[:top_k]

    def delete_embeddings(self, ids: List[str]) -> None:
        """
        Delete embeddings from the Chroma database.
        """
        self.collection.delete(ids=ids)
