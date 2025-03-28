import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import progressbar
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
                max_tokens=200,
            )
            self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
            self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
            self.summarize_msg = parse_txt(path(args.summarize_msg))
        else:
            raise ValueError(f"Invalid OpenAI model use case: {self.use_case}")

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, query):
        return self.model.embed_query(query)

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

        # bar = progressbar.ProgressBar(
        #     widgets=[
        #         "Summarizing texts: [",
        #         progressbar.Percentage(),
        #         "] ",
        #         progressbar.Bar(),
        #         " ",
        #         progressbar.ETA(),
        #     ],
        #     max_value=len(documents),
        # ).start()

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
                    doc.metadata["llm_summary"] = ""
                # bar.update(i + 1)
        # bar.finish()
