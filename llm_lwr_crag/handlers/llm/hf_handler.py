from typing import List

import torch
from langchain.schema import Document
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from utils.parse import parse_txt
from utils.path import path

from .abstract_llm import AbstractLLM


class HFHandler(AbstractLLM):
    def __init__(self, args):
        self.base_model = args.base_model
        self.device = "cuda" if args.device and torch.cuda.is_available() else "cpu"

        self.use_case = args.use_case
        if self.use_case == "embedding":
            self.model = HuggingFaceEmbeddings(
                model_name=self.base_model,
                model_kwargs={"device": self.device},
            )
        elif self.use_case == "generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

            # Setup a simple pipeline and create the wrapped model
            pipe = pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )
            self.model = HuggingFacePipeline(pipeline=pipe)

            # Load standard prompt templates
            self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
            self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
            self.summarize_msg = parse_txt(path(args.summarize_msg))
            self.augment_msg = parse_txt(path(args.augment_msg))
        elif self.use_case == "reranking":
            config = AutoConfig.from_pretrained(self.base_model)
            if not (
                config.architectures
                and any(
                    "ForSequenceClassification" in arch for arch in config.architectures
                )
            ):
                raise ValueError(
                    "Cannot perform reranking with a non-cross-encoder model."
                )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L6-v2"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Invalid Huggingface model use case: {self.use_case}")

    def __str__(self):
        return f"Huggingface({self.base_model})"

    def rerank(self, query: str, chunks: List[Document]) -> List[Document]:
        query_chunk_pairs = [
            [query for _ in range(len(chunks))],
            [ch.page_content for ch in chunks],
        ]
        feats = self.tokenizer(
            query_chunk_pairs[0],
            query_chunk_pairs[1],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        self.model.eval()
        with torch.no_grad():
            scores = self.model(**feats).logits

        chunk_score_pairs = list(zip(chunks, scores))
        sorted_chunk_score_pairs = sorted(
            chunk_score_pairs, key=lambda x: x[1], reverse=True
        )

        sorted_chunks = [pair[0] for pair in sorted_chunk_score_pairs]
        return sorted_chunks
