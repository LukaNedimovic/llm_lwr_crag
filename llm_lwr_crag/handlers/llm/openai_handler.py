from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import progressbar
from data_processing import ChunkDict
from langchain_openai import OpenAIEmbeddings

from .abstract_llm import AbstractLLM


class OpenAIHandler(AbstractLLM):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_threads = args.num_threads
        self.model_name = args.model_name
        self.model = OpenAIEmbeddings(
            model=self.model_name, openai_api_key=args.api_key
        )

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.model.embed_query(text), dtype=np.float32)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [
            (
                np.array(vec, dtype=np.float32)
                for vec in self.model.embed_documents(texts)
            )
        ]

    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        texts = []
        embeddings = []
        metadata = []
        ids = []

        chunk_texts = [chunk["text"] for chunk in chunks]
        text_batches = [
            chunk_texts[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(chunk_texts), self.batch_size)
        ]

        bar = progressbar.ProgressBar(
            widgets=[
                "Embedding chunks: [",
                progressbar.Percentage(),
                "] ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ],
            maxval=len(chunks),
        ).start()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            batch_embeddings = [
                emb
                for batch in executor.map(self._embed_batch, text_batches)
                for emb in batch
            ]

        index = 0
        for batch in batch_embeddings:
            for embedding in batch:
                chunk = chunks[index]

                texts.append(chunk["text"])
                embeddings.append(embedding)
                metadata.append({"source": chunk["source"], "chunk_index": index})
                ids.append(f"{chunk['source']}_{index}")

                bar.update(index + 1)
                index += 1
        bar.finish()

        return ChunkDict(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids,
        )
