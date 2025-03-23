from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from box import Box
from data_processing import ChunkDict
from langchain.schema import Document


class AbstractLLM(ABC):
    @abstractmethod
    def __init__(self, args: Box):
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text input.
        """
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Document]) -> ChunkDict:
        """
        Generate embeddings for a list of chunks (documents).
        """
        pass

    @abstractmethod
    def split_text(self, text: str) -> Optional[List[str]]:
        """
        Split given text into chunks.
        """
        pass

    @abstractmethod
    def gen_summary(self, text: str) -> Optional[str]:
        """
        Generate LLM summary for the given content.
        """
        pass

    @abstractmethod
    def gen_summaries(self, documents: List[Document]):
        """
        Generate LLM summary for the given content.
        """
        pass
