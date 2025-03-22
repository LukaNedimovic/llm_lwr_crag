from abc import ABC, abstractmethod
from typing import List

import numpy as np
from box import Box
from data_processing import ChunkDict


class AbstractLLM(ABC):
    @abstractmethod
    def __init__(self, args: Box):
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text input."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        """Generate embeddings for a list of chunks (documents)."""
        pass
