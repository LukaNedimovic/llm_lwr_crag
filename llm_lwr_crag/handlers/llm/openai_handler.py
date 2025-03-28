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
        elif self.use_case == "generation":
            self.model = ChatOpenAI(
                openai_api_key=args.api_key,
                model=self.model_name,
                temperature=0,
                max_tokens=200,
            )

            # Load standard prompt templates
            self.split_text_system_msg = parse_txt(path(args.split_text_system_msg))
            self.split_text_human_msg = parse_txt(path(args.split_text_human_msg))
            self.summarize_msg = parse_txt(path(args.summarize_msg))
            self.augment_msg = parse_txt(path(args.augment_msg))
        else:
            raise ValueError(f"Invalid OpenAI model use case: {self.use_case}")
