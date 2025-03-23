from typing import List, Literal, Optional

from pydantic import BaseModel, model_validator
from utils.const import DEFAULT_ARGS, LLM_SUMMARY_REQUIRED, REQUIRED_ARGS


class LLMConfig(BaseModel):
    """
    LLM YAML configuration validator.
    """

    provider: Literal["hf", "openai"] = DEFAULT_ARGS.retriever.llm.provider

    # Huggingface related arguments
    base_model: Optional[str] = DEFAULT_ARGS.retriever.llm.base_model
    device: Optional[str] = DEFAULT_ARGS.retriever.llm.device

    # OpenAI related arguments
    api_key: Optional[str] = DEFAULT_ARGS.retriever.llm.api_key
    model_name: Optional[str] = DEFAULT_ARGS.retriever.llm.model_name
    batch_size: Optional[int] = DEFAULT_ARGS.retriever.llm.batch_size
    num_threads: Optional[int] = DEFAULT_ARGS.retriever.llm.num_threads
    use_case: Literal["embedding", "chatting"] = DEFAULT_ARGS.retriever.llm.use_case
    split_text_system_msg: Optional[str] = (
        DEFAULT_ARGS.retriever.llm.split_text_system_msg
    )
    split_text_human_msg: Optional[str] = (
        DEFAULT_ARGS.retriever.llm.split_text_human_msg
    )
    summarize_msg: Optional[str] = DEFAULT_ARGS.retriever.llm.summarize_msg

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        retriever_llm_provider = values.get("provider")

        for required_arg in REQUIRED_ARGS.retriever.llm.provider[
            retriever_llm_provider
        ]:
            if not values.get(required_arg):
                raise ValueError(
                    (
                        f"`{required_arg}` is required when retriever is",
                        f"`{retriever_llm_provider}`.",
                    )
                )

        return values


class MetadataConfig(BaseModel):
    """
    Metadata-to-add prior to chunking YAML configuration validator.
    """

    list: List[str] = DEFAULT_ARGS.retriever.metadata.list  # type: ignore
    llm_summary: Optional[LLMConfig] = DEFAULT_ARGS.retriever.metadata.llm_summary

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        metadata_list = values.get("list")

        for md_pc in metadata_list:
            if md_pc in LLM_SUMMARY_REQUIRED and values.get("llm_summary") is None:
                raise ValueError(
                    (
                        "`LLM for summary is required for the metadata piece: ",
                        f"`{md_pc}`.",
                    )
                )

        return values


class RetrieverChunkingConfig(BaseModel):
    """
    LLM Chunking YAML configuration validator.
    """

    type: Literal[
        "RecursiveCharacterTextSplitter", "LLMChunking"
    ] = DEFAULT_ARGS.retriever.chunking.type  # type: ignore

    # RecursiveCharacterTextSplitter related arguments
    chunk_size: Optional[int] = DEFAULT_ARGS.retriever.chunking.chunk_size
    chunk_overlap: Optional[int] = DEFAULT_ARGS.retriever.chunking.chunk_overlap

    # LLMChunking related arguments
    llm_setup: Optional[LLMConfig] = DEFAULT_ARGS.retriever.chunking.llm_setup

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        chunking_type = values.get("type")

        for required_arg in REQUIRED_ARGS.retriever.chunking.type[chunking_type]:
            if not values.get(required_arg):
                raise ValueError(
                    (
                        f"`{required_arg}` is required when chunking type is",
                        f"`{chunking_type}`.",
                    )
                )

        return values


class RetrieverDBConfig(BaseModel):
    """
    Retriever (DB) YAML configuration validator.
    """

    provider: Literal["chromadb"] = DEFAULT_ARGS.retriever.db.provider  # type: ignore

    # ChromaDB related arguments
    chromadb_path: Optional[str] = DEFAULT_ARGS.retriever.db.chromadb_path
    collection_name: Optional[str] = DEFAULT_ARGS.retriever.db.collection_name

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        retriever_db_provider = values.get("provider")

        for required_arg in REQUIRED_ARGS.retriever.db.provider[retriever_db_provider]:
            if not values.get(required_arg):
                raise ValueError(
                    (
                        f"`{required_arg}` is required when database is",
                        f"`{retriever_db_provider}`.",
                    )
                )

        return values


class RetrieverConfig(BaseModel):
    """
    Retriever YAML configuration validator.
    """

    metadata: MetadataConfig
    chunking: RetrieverChunkingConfig
    db: RetrieverDBConfig
    llm: LLMConfig


class ConfigValidator(BaseModel):
    """
    Top-level YAML configuration validator.
    """

    mode: Literal["train"]  # type: ignore

    repo_url: Optional[str]
    repo_dir: Optional[str]
    eval_path: Optional[str]

    retriever: RetrieverConfig

    languages_path: Optional[str] = DEFAULT_ARGS.languages_path
    extensions_path: Optional[str] = DEFAULT_ARGS.extensions_path

    @model_validator(mode="before")
    def check_required_properties(cls, values):
        selected_mode = values.get("mode")

        for required_arg in REQUIRED_ARGS.mode[selected_mode]:
            if not values.get(required_arg):
                raise ValueError(
                    f"`{required_arg}` is required when mode is `{selected_mode}`."
                )

        return values
