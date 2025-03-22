from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import openai
import progressbar
from box import Box
from data_processing import ChunkDict

from .abstract_llm import AbstractLLM


class OpenAI(AbstractLLM):
    def __init__(self, args: Box):
        self.model = args.model
        self.client = openai.OpenAI(
            api_key=args.api_key
        )  # Initialize the OpenAI client
        self.batch_size = args.batch_size
        self.num_threads = args.num_threads

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text using the OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Sends a batch request to OpenAI API and returns embeddings."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        """
        Splits texts into batches, processes them in parallel, and returns a ChunkDict.
        """
        texts = []
        embeddings = []
        metadata = []
        ids = []

        # Extract texts for embedding
        chunk_texts = [chunk["text"] for chunk in chunks]
        text_batches = [
            chunk_texts[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(chunk_texts), self.batch_size)
        ]

        # Setup progress bar
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
            batch_embeddings = list(executor.map(self._embed_batch, text_batches))

        # Flatten the batch embeddings and store results
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
