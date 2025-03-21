from box import Box

__all__ = [
    "SUPPORTED_MODE",
    "SUPPORTED_DB",
    "SUPPORTED_RETRIEVER",
    "DEFAULT_ARGS",
    "REQUIRED_ARGS",
]

SUPPORTED_MODE = ["train"]
SUPPORTED_DB = ["chromadb"]
SUPPORTED_RETRIEVER = ["hf"]


DEFAULT_ARGS = Box(
    {
        "retriever": {
            "db": {
                "type": "chromadb",
                "chromadb_path": "$DB_DIR",
                "collection_name": "default_collection",
            },
            "llm": {
                "type": "hf",
                "base_model": "all-MiniLM-L6-v2",
                "device": "cuda",
            },
        }
    }
)

REQUIRED_ARGS = Box(
    {
        "mode": {
            "train": ["repo_url", "repo_dir", "eval_path"],
        },
        "retriever": {
            "db": {
                "type": {
                    "chromadb": ["chromadb_path"],
                }
            },
            "llm": {"type": {"hf": ["base_model", "device"]}},
        },
    }
)
