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

Since configuration was built to be modular, for complete set of arguments, please check: [Configuration README](./llm_lwr_crag/config/README.md).

To run a configuration from YAML file, simply provide it to the main file via the `--config` parameter:
```bash
./main.py --config path_to_config.yml
```

## 🚀 Quickstart
**LLM_LWR_CRAG** uses `conda` for environment management. To set up the environment, i.e. create it and install the dependencies, the setup script is provided:
```bash
source ./setup.sh
```
`setup.sh` will also export several environment variables, useful for dynamic path creation and, therefore, setting up the configurations for experiments.

Additionally, to be able to use **OpenAI** API, one must provide the key. It should be set either in the configuration YAML file, or within the `.env`. The latter is encouraged, to make configuration cleaner - check `.env.example`.

## 🧪 Experiments
**LLM_LWR_CRAG** comes with a set of 12 experiments.

Main goal was to perform a meaningful parameter sweep, and check for the most efficient configuration. You may find experiment results here: [Experiment Results Paper](./llm_lwr_crag/experiments/Experiment%20Results%20Paper.pdf).

## 📝 Documentation
To build the documentation, it is enough to run the `setup.sh` and the `build_docs.sh`:
```bash
source ./setup.sh
./build_docs.sh
```
By default, the `build_docs.sh` will open the `docs/build/index.html` using Firefox.
