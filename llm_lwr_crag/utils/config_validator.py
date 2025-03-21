from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from .const import DEFAULT_ARGS, REQUIRED_ARGS


class RetrieverConfig(BaseModel):
    """
    Retriever YAML configuration validator.
    """

    type: Literal["hf"] = DEFAULT_ARGS.retriever.type  # type: ignore

    # Huggingface related arguments
    base_model: Optional[str] = DEFAULT_ARGS.retriever.base_model
    device: Optional[str] = DEFAULT_ARGS.retriever.device

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        retriever_type = values.get("type")

        for required_arg in REQUIRED_ARGS.retriever.type[retriever_type]:
            if not values.get(required_arg):
                raise ValueError(
                    f"`{required_arg}` is required when retriever is `{retriever_type}`"
                )

        return values


class DatabaseConfig(BaseModel):
    """
    Database (DB) YAML configuration validator.
    """

    type: Literal["chromadb"] = DEFAULT_ARGS.db.type  # type: ignore

    # ChromaDB related arguments
    chromadb_path: Optional[str] = DEFAULT_ARGS.db.chromadb_path

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        db_type = values.get("type")

        for required_arg in REQUIRED_ARGS.db.type[db_type]:
            if not values.get(required_arg):
                raise ValueError(
                    f"`{required_arg}` is required when database is `{db_type}`"
                )

        return values


class ConfigValidator(BaseModel):
    """
    Top-level YAML configuration validator.
    """

    mode: Literal["train"]  # type: ignore

    repo_url: Optional[str]
    repo_dir: Optional[str]
    eval_path: Optional[str]

    db: DatabaseConfig
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
