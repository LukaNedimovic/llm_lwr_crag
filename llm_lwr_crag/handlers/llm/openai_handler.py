import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import progressbar
from data_processing import ChunkDict
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.parse import parse_txt
from utils.path import path

from .abstract_llm import AbstractLLM


class OpenAIHandler(AbstractLLM):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_threads = args.num_threads
        self.model_name = args.model_name
        self.use_case = args.use_case

        if self.use_case == "embedding":
            self.model = OpenAIEmbeddings(
                openai_api_key=args.api_key,
                model=self.model_name,
            )
        elif self.use_case == "chatting":
            self.model = ChatOpenAI(
                openai_api_key=args.api_key,
                model=self.model_name,
                temperature=0,
            )
            self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
            self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
            self.summarize_msg = parse_txt(path(args.summarize_msg))
        else:
            raise ValueError(f"Invalid OpenAI model use case: {self.use_case}")

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.model.embed_query(text), dtype=np.float32)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [
            (
                np.array(vec, dtype=np.float32)
                for vec in self.model.embed_documents(texts)
            )
        ]

    def embed_chunks(self, chunks: List[Document]) -> ChunkDict:
        texts = []
        embeddings = []
        metadata = []
        ids = []

        chunk_texts = [chunk.page_content for chunk in chunks]
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

                texts.append(chunk.page_content)
                embeddings.append(embedding)
                metadata.append(
                    {"source": chunk.metadata["source"], "chunk_index": index}
                )
                ids.append(f"{chunk.metadata['source']}_{index}")

                bar.update(index + 1)
                index += 1
        bar.finish()

        return ChunkDict(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids,
        )

    def split_text(self, text: str) -> List[str]:
        if self.use_case != "chatting":
            raise ValueError("Cannot perform splitting with a non-chat model.")

        prompt = [
            SystemMessage(content=self.split_text_system_msg),
            HumanMessage(
                content=f"""
                    {self.split_text_human_msg}

                    Text:
                    {text}
                    """
            ),
        ]

        json_llm = self.model.bind(response_format={"type": "json_object"})
        response = json_llm.invoke(prompt)

        try:
            result = json.loads(response.content)
            return result["chunks"]
        except json.JSONDecodeError:
            print("Error: LLM returned an invalid JSON format")
            return []

    def _split_text_batch(self, batch: List[str]) -> List[str]:
        chunks = []
        for text in batch:
            chunks.extend(self.split_text(text))
        return chunks

    def split_text_in_batches(self, texts: List[str]) -> List[str]:
        text_batches = [
            texts[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(texts), self.batch_size)
        ]

        bar = progressbar.ProgressBar(
            widgets=[
                "Splitting texts: [",
                progressbar.Percentage(),
                "] ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ],
            maxval=len(texts),
        ).start()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            all_chunks = list(executor.map(self._split_text_batch, text_batches))

        bar.finish()

        return [chunk for batch in all_chunks for chunk in batch]

    def gen_summary(self, text: str) -> Optional[str]:
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

    def gen_summaries(self, documents: List[Document]):
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
