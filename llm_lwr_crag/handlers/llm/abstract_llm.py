from abc import ABC, abstractmethod
from typing import List, Optional

from box import Box
from langchain.schema import Document


class AbstractLLM(ABC):
    @abstractmethod
    def __init__(self, args: Box):
        pass

    @abstractmethod
    def embed_query(self, query: str):
        """
        Embeds the query for the given retrieval task.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents):
        """
        Embeds the list of given documents.
        """
        pass

    @abstractmethod
    def split_text(self, text: str) -> Optional[List[str]]:
        """
        Split given text into chunks.
        """
        pass

    @abstractmethod
    def gen_summaries(self, documents: List[Document]):
        """
        Generate LLM summary for the given content.
        """
        pass
