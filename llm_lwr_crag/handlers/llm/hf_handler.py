from typing import List, Optional

import numpy as np
import progressbar
import torch
from data_processing import ChunkDict
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .abstract_llm import AbstractLLM


class HFHandler(AbstractLLM):
    def __init__(self, args):
        self.base_model = args.base_model

        self.device = "cuda" if args.device and torch.cuda.is_available() else "cpu"
        self.model = HuggingFaceEmbeddings(
            model_name=self.base_model,
            model_kwargs={"device": self.device},
        )

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.model.embed_query(text), dtype=np.float32)

    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        texts = []
        embeddings = []
        metadata = []
        ids = []

        # Create a progress bar using progressbar2
        bar = progressbar.ProgressBar(
            widgets=[
                "Embedding chunks: " "[",
                progressbar.Percentage(),
                "] ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ],
            maxval=len(chunks),
        ).start()

        # Generate embeddings for each chunk
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            source = chunk["source"]

            embedding = self.embed_text(text)

            chunk_id = f"{source}_{i}"
            texts.append(text)
            embeddings.append(embedding)
            metadata.append({"source": source, "chunk_index": i})
            ids.append(chunk_id)

            bar.update(i + 1)
        bar.finish()

        return ChunkDict(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids,
        )

    def split_text(self, text: str) -> Optional[ChunkDict]:
        """
        Chunk given text using a LLM.
        """
        pass

    def gen_summary(self, text: str) -> Optional[str]:
        """
        Generate LLM summary for the given content.
        """
        pass

    def gen_summaries(self, documents: List[Document]):
        """
        Generate LLM summary for the given content.
        """
        pass
