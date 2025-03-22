from box import Box

from .db import AbstractDBHandler, ChromaDBHandler
from .llm import AbstractLLMHandler, HFHandler

NAME_TO_DB_TYPE = {
    "chromadb": ChromaDBHandler,
}

NAME_TO_LLM_TYPE = {
    "hf": HFHandler,
}


class HandlerFactory:
    @staticmethod
    def get_db_handler(db_args: Box) -> AbstractDBHandler:
        """
        Factory method to return the appropriate database handler.
        """
        db_class = NAME_TO_DB_TYPE.get(db_args.type, None)
        if db_class is None:
            raise ValueError(f"Database type {db_args.type} is not supported.")

        db = db_class(db_args)
        return db

    @staticmethod
    def get_llm_handler(llm_args: Box) -> AbstractLLMHandler:
        """
        Factory method to return the appropriate LLM handler.
        """
        llm_class = NAME_TO_LLM_TYPE.get(llm_args.type, None)
        if llm_class is None:
            raise ValueError(f"LLM type {llm_args.type} is not supported.")

        llm = llm_class(llm_args)
        return llm
