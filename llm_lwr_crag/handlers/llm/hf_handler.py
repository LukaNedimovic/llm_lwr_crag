import torch
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
        self.device = "cuda" if args.device and torch.cuda.is_available() else "cpu"

        self.use_case = args.use_case
        if self.use_case == "embedding":
            self.model = HuggingFaceEmbeddings(
                model_name=self.base_model,
                model_kwargs={"device": self.device},
            )
        elif self.use_case == "generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)

            # Setup a simple pipeline and create the wrapped model
            pipe = pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
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
        else:
            raise ValueError(f"Invalid Huggingface model use case: {self.use_case}")
