from box import Box

from .db import AbstractDB, ChromaDBHandler
from .llm import AbstractLLM, HFHandler, OpenAIHandler

NAME_TO_DB_TYPE = {"chromadb": ChromaDBHandler}
NAME_TO_LLM_TYPE = {"hf": HFHandler, "openai": OpenAIHandler}


class AutoDB:
    @staticmethod
    def from_args(db_args: Box) -> AbstractDB:
        """
        Factory method to return the appropriate database handler.
        """
        db_class = NAME_TO_DB_TYPE.get(db_args.provider, None)
        if db_class is None:
            raise ValueError(f"Database type {db_args.provider} is not supported.")

        db = db_class(db_args)
        return db


class AutoLLM:
    @staticmethod
    def from_args(llm_args: Box) -> AbstractLLM:
        """
        Factory method to return the appropriate LLM handler.
        """
        llm_class = NAME_TO_LLM_TYPE.get(llm_args.provider, None)
        if llm_class is None:
            raise ValueError(f"LLM type {llm_args.provider} is not supported.")

        llm = llm_class(llm_args)
        return llm
