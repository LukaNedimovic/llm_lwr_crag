from abc import ABC
from typing import List, Optional, Union

from box import Box
from langchain.schema import Document, HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel


class AbstractLLM(ABC):
    use_case: str
    model: Union[BaseLanguageModel, Embeddings]
    summarize_msg: Optional[str] = ""

    def __init__(self, args: Box):
        pass

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds the query for the given retrieval task.
        """
        if self.use_case != "embedding":
            raise ValueError("Cannot embed the query using non-embedding model.")
        return self.model.embed_query(query)

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """
        Embeds the list of given documents.
        """
        if self.use_case != "embedding":
            raise ValueError("Cannot embed documents using non-embedding model.")
        return self.model.embed_documents(docs)

    def gen_summary(self, doc_or_text: Union[Document, str]) -> str:
        """
        Generate LLM summary for the given content.
        """
        if self.use_case != "generation":
            raise ValueError("Cannot generate summary using non-generative model.")

        # If Document is provided
        text = doc_or_text
        if isinstance(text, Document):
            text = text.page_content

        prompt = [
            HumanMessage(
                content=f"""
                    {self.summarize_msg}

                    Text:
                    {text}
                    """
            ),
        ]
        response = self.model.invoke(prompt)
        return response.content.strip()
