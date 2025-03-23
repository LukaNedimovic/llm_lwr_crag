from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import progressbar
import torch
from data_processing import ChunkDict
from langchain.schema import Document, HumanMessage
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from utils.parse import parse_txt
from utils.path import path

from .abstract_llm import AbstractLLM


class HFHandler(AbstractLLM):
    def __init__(self, args):
        self.base_model = args.base_model
        self.num_threads = args.num_threads
        self.batch_size = args.batch_size

        self.device = "cuda" if args.device and torch.cuda.is_available() else "cpu"

        self.use_case = args.use_case
        if self.use_case == "embedding":
            self.model = HuggingFaceEmbeddings(
                model_name=self.base_model,
                model_kwargs={"device": self.device},
            )
        elif self.use_case == "chatting":
            model_id = "google/flan-t5-small"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            pipe = pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                truncation=True,
                do_sample=False,
                repetition_penalty=1.03,
            )
            self.model = HuggingFacePipeline(pipeline=pipe)

            self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
            self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
            self.summarize_msg = parse_txt(path(args.summarize_msg))
        else:
            raise ValueError(f"Invalid OpenAI model use case: {self.use_case}")

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.model.embed_query(text), dtype=np.float32)

    def embed_chunks(self, chunks: List[Document]) -> ChunkDict:
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
            text = chunk.page_content
            source = chunk.metadata["source"]

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
        raise RuntimeError("split_text() currently unsupported for HF.")

    def gen_summary(self, text: str) -> Optional[str]:
        """
        Generate LLM summary for the given content.
        """
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
        return response

    def gen_summaries(self, documents: List[Document]):
        """
        Generate LLM summary for the given content.
        """
        if self.use_case != "chatting":
            raise ValueError("Cannot perform summarization with a non-chat model.")

        bar = progressbar.ProgressBar(
            widgets=[
                "Summarizing texts: [",
                progressbar.Percentage(),
                "] ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ],
            max_value=len(documents),
        ).start()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_document = {
                executor.submit(self.gen_summary, doc.page_content): doc
                for doc in documents
            }

            for i, future in enumerate(as_completed(future_to_document)):
                doc = future_to_document[future]
                try:
                    summary = future.result()
                    doc.metadata["llm_summary"] = summary
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                bar.update(i + 1)
        bar.finish()
