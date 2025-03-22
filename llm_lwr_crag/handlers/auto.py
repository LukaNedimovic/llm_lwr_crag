from box import Box

from .db import AbstractDB, ChromaDB
from .llm import HF, AbstractLLM

NAME_TO_DB_TYPE = {"chromadb": ChromaDB}


class AutoDB:
    @staticmethod
    def from_args(db_args: Box) -> AbstractDB:
        """
        Factory method to return the appropriate database handler.
        """
        db_class = NAME_TO_DB_TYPE.get(db_args.type, None)
        if db_class is None:
            raise ValueError(f"Database type {db_args.type} is not supported.")

        db = db_class(db_args)
        return db


NAME_TO_LLM_TYPE = {"hf": HF}


class AutoLLM:
    @staticmethod
    def from_args(llm_args: Box) -> AbstractLLM:
        """
        Factory method to return the appropriate LLM handler.
        """
        llm_class = NAME_TO_LLM_TYPE.get(llm_args.type, None)
        if llm_class is None:
            raise ValueError(f"LLM type {llm_args.type} is not supported.")

        llm = llm_class(llm_args)
        return llm
