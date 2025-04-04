#!/usr/bin/env python3

import os

import pipeline as pl
from box import Box  # type: ignore
from dotenv import load_dotenv
from langchain.globals import get_verbose, set_verbose
from rag import RAG
from utils import path  # For variable expansion
from utils import log_res, logger, parse_args

# Load .env file, located in DOTENV_PATH env variable
load_dotenv(path(os.environ.get("DOTENV_PATH")))

# Turn off Langchain verbose mode
set_verbose(False)
is_verbose = get_verbose()


def train(args: Box) -> None:
    """
    Perform training and evaluation.
    General training process:
        (1) Download the GitHub directory
        (2) Load the evaluation dataset
        (3) Generate extensions, or load them if already present
        (4) Load documents, alongside their designated metadata
        (5) Chunk the documents
        (6) Set up retrieval embedding function (generally, an embedding LLM)
        (7) Set up retrieval database, using the aforementioned emb. function
        (8) Add documents to the database
        (9) Evaluate

    Args:
        args (Box): Parsed YML file configuration arguments.
            Will be used in several setup steps.

    Returns:
        None
    """
    # Download repo and parse evaluation data
    # Repo is not being returned since it is downloaded to local
    # Evaluation dataset is wrapped into a `pandas.DataFrame` object
    eval_df = pl.make_repo_and_eval(args)
    docs, chunks = pl.load_docs_and_chunk(args)  # Load and chunk documents
    rag = RAG.from_args(args, docs, chunks)  # Set up RAG with given docs / chunks

    # Evaluate RAG on dataset
    # log_path is the place of the local log file
    avg_recall = rag.eval(eval_df, k=args.retriever.k)

    # Log if applicable
    log_res(
        log_path=path(args.log_path),
        log_dict={
            "exp_name": args.exp_name,
            "eval": args.retriever.get("eval", None),
            "ret_chunker": args.retriever.chunking.type,
            "chunk_size": args.retriever.chunking.chunk_size,
            "chunk_overlap": args.retriever.chunking.chunk_overlap,
            "num_docs": str(len(docs)),
            "num_chunks": str(len(chunks)),
            "metadata": args.retriever.metadata,
            "ret_vec_db": str(rag.retriever.vec_db),
            "ret_db_bm25": str(rag.retriever.bm25),
            "ret_rerank": str(rag.retriever.rerank),
            "gen_llm": str(rag.generator.llm),
            "avg_recall": avg_recall,
            "k": args.retriever.k,
        },
    )

    logger.info(f"{avg_recall * 100:.2f}")



import gradio as gr

def test_load_github_repo(url):
    """Simulate loading a GitHub repository (for now, just returns confirmation)."""
    return f"✅ Loaded GitHub Repo: {url}"

def test_process_query(provider, model, query):
    """Simulate an AI processing the query."""
    if not provider or not model:
        return "⚠️ Please enter an AI provider and model name."
    return f"🔍 Processing query '{query}' using {provider} - {model}..."

def train_ui():
    with gr.Blocks(css="""
        #load_button {
            width: 60px !important;
            height: 100% !important; /* Fills container */
            min-width: 60px !important;
            max-width: 60px !important;
            min-height: 60px !important;
            max-height: 100% !important;
            font-size: 14px !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex-shrink: 0; /* Prevents shrinking */
        }      
        
        /* URL input - now visible and properly sized */
        #github_url {
            border: none !important;
            box-shadow: none !important;
            flex-grow: 1;
            padding: 12px !important;
            background: transparent !important;
        }
        
        /* Centered title */
        .title {
            text-align: center !important;
            width: 100% !important;
            margin-bottom: 20px !important;
        }
        
        /* Main container */
        .main-container {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding-bottom: 80px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Query input - pure and centered */
        #query_input {
            position: fixed !important;
            bottom: 20px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: 25% !important;
            min-width: 300px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 1px solid var(--border-color-primary) !important;
            border-radius: 4px !important;
        }
    """) as app:
        with gr.Column(elem_classes="main-container"):
            # Centered title
            gr.Markdown("## 🤖 LLM Listwise Reranker for CodeRAG", elem_classes="title")
            
            # URL row with Load button
            with gr.Row(elem_classes="url-row"):
                with gr.Column(elem_classes="url-input-container"):
                    github_url = gr.Textbox(label="GitHub Repository URL")
                load_button = gr.Button("Load", elem_id="load_button")
            
            gr.Markdown("### Retrieval")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Chunking Strategy")
                    with gr.Row():
                        chunk_size = gr.Textbox(label="Chunk Size", interactive=True)
                        chunk_overlap = gr.Textbox(label="Chunk Overlap", interactive=True)
                
                with gr.Column():
                    gr.Markdown("#### Embedding Model")
                    with gr.Row():
                        emb_model_provider = gr.Textbox(label="Embedding Model Provider", interactive=True)
                        emb_model_name = gr.Textbox(label="Embeding Model Name", interactive=True)
                        
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Metadata")    
                    with gr.Row():
                        code_structure = gr.Checkbox(label="Code Structure")
                    with gr.Row():
                        llm_summary_provider = gr.Textbox(label="Summary LLM Provider", interactive=True)
                        llm_summary_model_name = gr.Textbox(label="Summary LLM Model Name", interactive=True)
                        
                with gr.Column():
                    gr.Markdown("#### Database")    
                    with gr.Row():
                        db = gr.Dropdown(label="Database", choices=["ChromaDB", "FAISS"], interactive=True)

            gr.HTML("<hr>")
            
            query_output = gr.Textbox(label="Output", interactive=False)
            
            # Pure query input
            query_input = gr.Textbox(
                elem_id="query_input",
                placeholder="Type your query...",
                show_label=False,
                container=False
            )

        # load_button.click(test_load_github_repo, inputs=github_url, outputs=query_output)
        # query_input.submit(test_process_query, inputs=[ai_provider, model_name, query_input], outputs=query_output)

    app.launch()
    

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        if args.ui:
            train_ui()
        else:
            train(args)