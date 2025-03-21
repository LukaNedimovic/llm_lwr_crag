from typing import Any, List

from .abstract_llm_handler import AbstractLLMHandler


class HFHandler(AbstractLLMHandler):
    def __init__(self, **kwargs):
        pass

    def create_embeddings(self, documents: List[str]) -> List[Any]:
        """Generate embeddings for a list of documents."""
        return ["TODO"]
