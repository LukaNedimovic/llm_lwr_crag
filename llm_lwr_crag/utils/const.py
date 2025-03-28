from box import Box

__all__ = [
    "SUPPORTED_MODE",
    "SUPPORTED_DB",
    "SUPPORTED_RETRIEVER_LLM",
    "DEFAULT_ARGS",
    "REQUIRED_ARGS",
    "LLM_SUMMARY_REQUIRED",
]

SUPPORTED_MODE = ["train"]
SUPPORTED_DB = ["chromadb"]
SUPPORTED_RETRIEVER_LLM = ["hf", "openai"]

DEFAULT_ARGS = Box(
    {
        "retriever": {
            "eval": {
                "augment_query": None,
            },
            "metadata": {
                "list": [],
                "llm_summary": None,
            },
            "chunking": {
                "type": "RecursiveCharacterTextSplitter",
                "metadata": [],
                "chunk_size": 500,
                "chunk_overlap": 50,
                "llm_setup": None,
            },
            "db": {
                # ChromaDB
                "provider": "chromadb",
                "collection_name": "default_collection",
                "chromadb_path": "$PERSIST_DIR/chroma/",
                "faiss_path": "$PERSIST_DIR/faiss/",
            },
            "llm": {
                "provider": "hf",
                # General API
                "api_key": "your-api-key",
                # Huggingface
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda",
                # OpenAI
                "model_name": "text-embedding-ada-002",
                "batch_size": 16,
                "num_threads": 12,
                "use_case": "embedding",
                "split_text_system_msg": "$PROMPTS_DIR/split_text_sys_default.txt",
                "split_text_human_msg": "$PROMPTS_DIR/split_text_hmn_default.txt",
                "summarize_msg": "$PROMPTS_DIR/summarize_msg.txt",
                "augment_msg": "$PROMPTS_DIR/augment_msg.txt",
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
                    "LLMChunking": ["llm_setup"],
                },
            },
            "db": {
                "provider": {
                    "chromadb": ["chromadb_path"],
                    "faiss": ["faiss_path"],
                }
            },
            "llm": {
                "provider": {
                    "hf": ["base_model", "device"],
                    "openai": ["api_key", "model_name"],
                    "google": ["api_key", "model_name"],
                },
            },
        },
    }
)

LLM_SUMMARY_REQUIRED = ["llm_summary"]
