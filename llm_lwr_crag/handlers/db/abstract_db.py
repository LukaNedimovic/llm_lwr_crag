from abc import ABC, abstractmethod
from typing import Any, List


class AbstractDB(ABC):
    """
    Abstract handler class for seamless integration with various databases.
    """

    @abstractmethod
    def store_embeddings(self, chunks_data: dict) -> None:
        """
        Store embeddings in the database.
        """
        pass

    @abstractmethod
    def query(self, query_embedding: Any, top_k: int) -> List[str]:
        """
        Query the database to retrieve the most similar documents.
        """
        pass

    @abstractmethod
    def delete_embeddings(self, ids: List[str]) -> None:
        """
        Delete embeddings from the database by IDs.
        """
        pass
