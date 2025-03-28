from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractDB(ABC):
    """
    Abstract handler class for seamless integration with various databases.
    """

    @abstractmethod
    def add_documents(self, chunks) -> None:
        """
        Store embeddings in the Chroma database.
        """
        pass

    @abstractmethod
    def query(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the Chroma database for files.
        This modified version returns both the file paths and their similarity scores.
        """
        pass
