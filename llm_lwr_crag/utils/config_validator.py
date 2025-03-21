from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from .const import DEFAULT_ARGS, REQUIRED_ARGS


class RetrieverLLMConfig(BaseModel):
    """
    LLM Retriever YAML configuration validator.
    """

    type: Literal["hf"] = DEFAULT_ARGS.retriever.llm.type  # type: ignore

    # Huggingface related arguments
    base_model: Optional[str] = DEFAULT_ARGS.retriever.llm.base_model
    device: Optional[str] = DEFAULT_ARGS.retriever.llm.device

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        retriever_llm_type = values.get("type")

        for required_arg in REQUIRED_ARGS.retriever.llm.type[retriever_llm_type]:
            if not values.get(required_arg):
                raise ValueError(
                    (
                        f"`{required_arg}` is required when retriever is",
                        f"`{retriever_llm_type}`",
                    )
                )

        return values


class RetrieverDBConfig(BaseModel):
    """
    Retriever (DB) YAML configuration validator.
    """

    type: Literal["chromadb"] = DEFAULT_ARGS.retriever.db.type  # type: ignore

    # ChromaDB related arguments
    chromadb_path: Optional[str] = DEFAULT_ARGS.retriever.db.chromadb_path
    collection_name: Optional[str] = DEFAULT_ARGS.retriever.db.collection_name

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        retriever_db_type = values.get("type")

        for required_arg in REQUIRED_ARGS.retriever.db.type[retriever_db_type]:
            if not values.get(required_arg):
                raise ValueError(
                    (
                        f"`{required_arg}` is required when database is",
                        f"`{retriever_db_type}`",
                    )
                )

        return values


class RetrieverConfig(BaseModel):
    """
    Retriever YAML configuration validator.
    """

    db: RetrieverDBConfig
    llm: RetrieverLLMConfig


class ConfigValidator(BaseModel):
    """
    Top-level YAML configuration validator.
    """

    mode: Literal["train"]  # type: ignore

    repo_url: Optional[str]
    repo_dir: Optional[str]
    eval_path: Optional[str]

    retriever: RetrieverConfig

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        selected_mode = values.get("mode")

        for required_arg in REQUIRED_ARGS.mode[selected_mode]:
            if not values.get(required_arg):
                raise ValueError(
                    f"`{required_arg}` is required when mode is `{selected_mode}`"
                )

        return values
