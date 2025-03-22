from abc import ABC, abstractmethod
from typing import List


class AbstractLLM(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[dict]) -> dict:
        """Generate embeddings for a list of chunks (documents)."""
        pass
