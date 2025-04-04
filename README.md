# 🤖 LLM Listwise Reranker for CodeRAG

**LLM_LWR_CRAG** is a RAG implementation, with high degree of customization.

Originally implemented as a part of JetBrains Internship Application Test Task.


## 🏗 Implementation
### ℹ️ General
**LLM_LWR_CRAG** aims to create a RAG framework capable of downloading a **GitHub** repository, given the URL, and perform file-retrieval tasks from the user query. As a part of the test task, the project comes with the evaluation dataset, and is evaluated solely on: [Escrcpy GitHub Repository](https://github.com/viarotel-org/escrcpy), with **Recall@10** being the metric.

### 🖇 Handlers
The project is structured around database and LLM handlers (`AbstractDB` & `AbstractLLM`), capable of providing seamless integratino with variuos APIs / frameworks. Currently, we support:
 - Databases:
   - `ChromaDB`
   - `FAISS`
   - `BM25` (for hybrid search)
 - LLMs (providers):
   - `OpenAI`
   - `Huggingface`

By implementing methods from parents classes, it is easy to integrate new technologies into the framework.

### 🔧 Configuration
To run the experiments, we provide `ConfigValidator` - `pydantic`-based YAML configuration validator, that was chosen instead of `argparse`, for better modularity and precise argument control.
Main parts of the configuration are `retriever` and `generator`, responsible for setting up `RAG` pipeline. Other, general arguments, can be provided aside from these two key parts.

In the table below, you may find the overview of possible arguments:

| Argument Name                           | Description | Value Range   | Default Value |
|-----------------------------------------|-------------|---------------|---------------|
| exp_name                                | Experiment name |  | |
| mode                                    | Mode to run the program in | `train` (for training / evaluation) | `train`|
| repo_url                                | GitHub repository URL| Any accessible GitHub repository. | https://github.com/viarotel-org/escrcpy |
| repo_dir                                | Directory to download the repository to |               | |
| eval_path                               | Path to evaluation dataset            |               | |
| retriever                               | Complete retriever setup            |               | |
| &nbsp;eval                              |             |               | |
| &nbsp;&nbsp;augment_query (`LLMConfig`) | Query augmentation            |               | |
| &nbsp;metadata                          | Metadata generation            |               | |
| &nbsp;chunking                          | Chunking strategy            |               | |
| &nbsp;db                                | (Vector) database             |               | |
| &nbsp;llm (`LLMConfig`)                 | Embedding model            |               | |
| &nbsp;rerank (`LLMConfig`)              | Reranker             |               | |
| generator (`LLMConfig`)                 | Generator LLM            |               | |
| languages_path                          | Path to languages file     |               | |
| extensions_path                         | Path to save generated extensions to             |               | |

## 🚀 Quickstart
**LLM_LWR_CRAG** uses `conda` for environment management. To set up the environment, i.e. create it and install the dependencies, the setup script is provided:
```bash
source ./setup.sh
```
`setup.sh` will also export several environment variables, useful for dynamic path creation and, therefore, setting up the configurations for experiments.

Additionally, to be able to use **OpenAI** API, one must provide the key. It should be set either in the configuration YAML file, or within the `.env`. The latter is encouraged, to make configuration cleaner - check `.env.example`.

## 🧪 Experiments
**LLM_LWR_CRAG** comes with a set of 13 experiments. The main goal was to perform a meaningful parameter sweep, and check for the most efficient configuration.


## 📝 Documentation
To build the documentation, it is enough to run the `setup.sh` and the `build_docs.sh`:
```bash
source ./setup.sh
./build_docs.sh
```
By default, the `build_docs.sh` will open the `docs/build/index.html` using Firefox.
