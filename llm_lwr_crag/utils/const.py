from box import Box

__all__ = [
    "SUPPORTED_MODE",
    "SUPPORTED_DB",
    "SUPPORTED_RETRIEVER_LLM",
    "DEFAULT_ARGS",
    "REQUIRED_ARGS",
]

SUPPORTED_MODE = ["train"]
SUPPORTED_DB = ["chromadb"]
SUPPORTED_RETRIEVER_LLM = ["hf", "openai"]

DEFAULT_ARGS = Box(
    {
        "retriever": {
            "chunking": {
                "type": "RecursiveCharacterTextSplitter",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "db": {
                # ChromaDB
                "type": "chromadb",
                "chromadb_path": "$PERSIST_DIR/chroma/",
                "collection_name": "default_collection",
            },
            "llm": {
                "type": "hf",
                # Huggingface
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda",
                # OpenAI
                "model": "text-embedding-ada-002",
                "batch_size": 16,
                "num_threads": 12,
            },
        },
        "languages_path": "$DATA_DIR/languages.yml",
        "extensions_path": "$DATA_DIR/extensions.txt",
    }
)

REQUIRED_ARGS = Box(
    {
        "mode": {
            "train": ["repo_url", "repo_dir", "eval_path"],
        },
        "retriever": {
            "chunking": {
                "type": {
                    "RecursiveCharacterTextSplitter": [],
                },
            },
            "db": {
                "type": {
                    "chromadb": ["chromadb_path"],
                }
            },
            "llm": {
                "type": {
                    "hf": ["base_model", "device"],
                    "openai": ["api_key", "model"],
                },
            },
        },
    }
)
