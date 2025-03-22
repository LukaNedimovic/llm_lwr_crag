from abc import ABC, abstractmethod
from typing import List

from data_processing import ChunkDict


class AbstractLLM(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        """Generate embeddings for a list of chunks (documents)."""
        pass
