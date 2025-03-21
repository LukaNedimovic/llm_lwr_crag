from abc import ABC, abstractmethod
from typing import Any, List


class AbstractLLMHandler(ABC):
    @abstractmethod
    def create_embeddings(self, documents: List[str]) -> List[Any]:
        """Generate embeddings for a list of documents."""
        pass
