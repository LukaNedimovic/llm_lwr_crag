from typing import List

import progressbar
import torch
from data_processing import ChunkDict
from transformers import AutoModel, AutoTokenizer

from .abstract_llm import AbstractLLM


class HF(AbstractLLM):
    def __init__(self, args):
        self.base_model = args.base_model

        # Set up tokenizer and model itself
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModel.from_pretrained(self.base_model)

        # Move model to adequate device
        self.device = "cuda" if args.device and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.squeeze().detach().cpu().numpy()

    def embed_chunks(self, chunks: List[dict]) -> ChunkDict:
        """Generate embeddings for a list of chunks (documents)."""
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
