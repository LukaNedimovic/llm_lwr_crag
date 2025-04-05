import os
from typing import List

from langchain.schema import Document, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.parse import parse_txt
from utils.path import path

from .abstract_llm import AbstractLLM


class OpenAIHandler(AbstractLLM):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_threads = args.num_threads
        self.model_name = args.model_name

        # Load standard prompt templates
        self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
        self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
        self.summarize_msg = parse_txt(path(args.summarize_msg))
        self.augment_msg = parse_txt(path(args.augment_msg))
        self.rerank_msg = parse_txt(path(args.rerank_msg))
        self.generate_msg = parse_txt(path(args.generate_msg))

        # Set up api key
        self.api_key = args.api_key
        if args.api_key is None:
            env_key = os.environ.get("OPENAI_API_KEY", None)
            if env_key:
                args.api_key = env_key
            else:
                raise ValueError(
                    "OPENAI_API_KEY not provided in .env, nor in config files"
                )

        self.use_case = args.use_case
        if self.use_case == "embedding":
            self.model = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model=self.model_name,
            )
        elif self.use_case == "generation" or self.use_case == "reranking":
            self.model = ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model_name,
                temperature=0,
                max_tokens=200,
            )
        else:
            raise ValueError(f"Invalid OpenAI model use case: {self.use_case}")

    def __str__(self):
        return f"OpenAI({self.model_name})"

    def rerank(self, query: str, chunks: List[Document]) -> List[Document]:
        scored_chunks = []
        for chunk in chunks:
            prompt = [
                HumanMessage(
                    content=f"""
                        {self.rerank_msg}

                        Query:
                        {query}

                        Chunk:
                        {chunk}
                        """
                ),
            ]

            response = self.model.invoke(prompt)
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 0.0  # Default score if parsing fails
            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]

    def generate(self, query: str, chunks: List[Document]) -> str:
        concat_chunks = "\n\n".join([ch.page_content for ch in chunks])
        prompt = [
            HumanMessage(
                content=f"""
                    {self.generate_msg}

                    Query:
                    {query}

                    Chunks (used for answer generation)
                    {concat_chunks}
                    """
            ),
        ]
        response = self.model.invoke(prompt)
        return response.content.strip()
