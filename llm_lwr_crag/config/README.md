# 🔧 Configuration

## 🗺️ Overview
Configuration is implemented using YAML files. The general parts of configuration consist of several separate classes, that can be easily modified and reused.

In the table below, you may find the top-level overview of possible arguments:

| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| exp_name                                | Experiment name |  | |
| mode                                    | Mode to run the program in | `train` (for training / evaluation) | `train`|
| repo_url                                | GitHub repository URL| Any accessible GitHub repository. | https://github.com/viarotel-org/escrcpy |
| repo_dir                                | Directory to download the repository to |               | |
| eval_path                               | Path to evaluation dataset            |               | |
| retriever                               | Complete retriever setup            |               | |
| &nbsp;eval (`EvalConfig`)                             |             |               | |
| &nbsp;metadata (`MetadataConfig`)                         | Metadata generation            |               | |
| &nbsp;chunking (`ChunkingConfig`)                         | Chunking strategy            |               | |
| &nbsp;db (`DBConfig`)                                | (Vector) database             |               | |
| &nbsp;llm (`LLMConfig`)                 | Embedding model            |               | |
| &nbsp;rerank (`LLMConfig`)              | Reranker             |               | |
| generator (`LLMConfig`)                 | Generator LLM            |               | |
| languages_path                          | Path to languages file     |               | |
| extensions_path                         | Path to save generated extensions to             |               | |

Please check other specific modules of configuration, for detailed explanation of capabilities.

### 🈹 `LLMConfig`

`LLMConfig` is used to set up an LLM for a variety of tasks, such as embedding (e.g. for vector database) or text generation (e.g. answering the query).

| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| provider | Provider to use for LLM access | "openai", "hf" | "hf"|
| device                                 | (HF) Device to host the model on |  "cpu", "cuda"| "cuda" |
| api_key | (OAI) API key |  | `None` |
| model_name | Name of the model from given provider |  | `sentence-transformers/all-MiniLM-L6-v2` |
| batch_size* | Batch size for embedding | `int` | 32 |
| num_threads* | Number of workers to assign for the task  | `int` | 16 |
| use_case | Use case of the model | "embedding", "generation", "reranking" | "embedding" |
| split_text_system_msg* | Path to `.txt` file containing system message for LLM, in text chunking task |  | `None` |
| split_test_human_msg* | Path to `.txt` file containing human message for LLM, in text chunking task |  | `None` |
| summarize_msg | Path to `.txt` file containing message for LLM (document summarization task) |  | `None` |
| augment_msg | Path to `.txt` file containing message for LLM, (document summarization task) |  | `None` |
| rerank_msg | Path to `.txt` file containing message for LLM, (document reranking task) |  | `None` |
| generate_msg | Path to `.txt` file containing message for LLM, (text generation task) |  | `None` |
> **Note:** Arguments marked with `*` are left for demonstration purposes.

### 💾 `DBConfig`

`DBConfig` is used to configure the (vector) database, used for embedding storage.


| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| provider | Database provider (kind) | "chromadb", "faiss" | "chromadb" |
| collection_name | Name of the collection to create | `str` | "default_collection" |
| chromadb_path | Path to store the ChromaDB database |  | `$PERSIST_DIR/chroma/` |
| faiss_path | Path to store the FAISS database |  | `$PERSIST_DIR/faiss/` |

### 🏷️ `MetadataConfig`

`MetadataConfig` is used to configure the pieces of extra metadata to be appended to the files. As of now, metadata is also being appended to the chunk's / document's content. Also, it is added to the `langchain.Document.metadata` object directly.

| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| list | List of pieces of metadata to append. Pieces are represented in a list, such as `["llm_summary"]` | "code_structure", "llm_sumary" (list) | [] |
| llm_summary (`LLMConfig`) | Configuration for the LLM used to generate document summary. Must be provided if `llm_summary` is selected as a metadata piece. |  | `None` |

### 🧩 `ChunkingConfig`

`ChunkingConfig` is used to describe the way the document ought to be chunked.

| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| type | Chunking method to use | "RecursiveCharacterTextSplitter", "LLMChunking" | "RecursiveCharacterTextSplitter" |
| chunk_size | (RCTS) Maximum size of a single chunk | int | 500 |
| chunk_overlap | (RCTS) Overlap between two adjacent chunks | int | 50 |
| llm_setup (`LLMConfig`) | (LLMChunking) Configuration of the chunking LLM. Must be provided if "LLMChunking" is selected |  | `None` |

### 🔍 `EvalConfig`
`EvalConfig` is used to configure part of evaluation dataset. As of now, it only supports augmenting queries.
| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| augment_query (`LLMConfig`) | Configuration for the LLM used to augment queries |  | `None` |

## 💡 Example

Examples of configuration can be found in [`experiments directory`](../experiments/).
