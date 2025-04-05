import os
from functools import partial
from typing import Generator, Tuple

import gradio as gr
import utils.pipeline as pl
from box import Box
from rag import RAG
from utils import download_repo, path

rag = None


def load_css():
    """
    Load gradio CSS file.

    Returns:
        Parsed CSS content.
    """
    with open(os.environ.get("GRADIO_CSS_PATH", None)) as file:
        css_content = file.read()
    return css_content


def ui__download_repo_and_setup_rag(github_url, args: Box):
    """
    Download repository and set up RAG for UI mode.
    Notice: This method does not load evaluation dataset, since there is no
    evaluation happening at all - the user is interacting with the system.

    Args:
        github_url: URL to GitHub repository to clone.
        args (Box): Parsed YAML configuration arguments.

    Returns:
        None
    """
    # Download GitHub repo and update the status bar accordingly
    download_repo_gen = download_repo(
        github_url,
        repo_dir=path(args.repo_dir),
        force_download=False,
    )
    yield from download_repo_gen

    yield "⌛ Loading and chunking documents..."
    docs, chunks = pl.load_docs_and_chunk(args)
    yield "✅ Documents successfully chunked!"

    yield "⌛ Building RAG system..."
    global rag
    rag = RAG.from_args(args, docs, chunks)  # Set up RAG with given docs / chunks
    yield "✅ Sucessfully built RAG system!"


def ui__query(query: str, k: int = 10) -> Generator[Tuple[str, str, str], None, None]:
    """
    (UI) Retrieve top-K relevant file paths and generated answer, for given query.

    Args:
        query (str): Query to pass to the system.
        k (int): Top-K parameter

    Returns:
        Tuple[str, str, str]: A tuple consisting of:
            (1) Status update message
            (2) ret_fps_output (str): Concatenated list of file paths
            (3) gen_ans (str): Generated answer
    """
    if rag is None:
        yield "❌ Cannot query the system prior to its instruction.", "", ""
        return

    yield "⌛ Querying...", "", ""
    ret_fps, _, gen_ans = rag(query, k)

    # Format the output
    ret_fps_output = "\n".join(ret_fps)
    if gen_ans is None:
        gen_ans = ""

    yield "✅ Sucessful!", ret_fps_output, gen_ans


def setup_ui(args: Box):
    """
    Set up interactive UI using Gradle.

    Args:
        args (Box): Parsed YAML configuration arguments.

    Returns:
        None
    """
    with gr.Blocks(css=load_css()) as app:
        with gr.Column(elem_classes="main-container"):
            # Centered title
            gr.Markdown("## 🤖 LLM Listwise Reranker for CodeRAG", elem_classes="title")

            with gr.Row():
                # GitHub URL input field
                # Pass the `utils.download_repo` as partial function,
                # containing the relevant kwargs from YAML configuration
                # `github_url` input field is going to be passed
                # as the first argument
                status_field = gr.Textbox(
                    label="Status",
                    value="Waiting for input...",
                    elem_id="status-field",
                    interactive=False,
                )

            with gr.Row():
                github_url = gr.Textbox(
                    label="GitHub Repository URL", elem_id="github_url"
                )
                github_url.submit(
                    partial(
                        ui__download_repo_and_setup_rag,
                        args=args,
                    ),
                    inputs=[github_url],
                    outputs=[status_field],
                )

            gr.HTML("<hr>")

            with gr.Column():
                gr.Markdown("#### 📝 Output")

                ret_fps_output = gr.Textbox(
                    label="Retrieved files",
                    placeholder="Waiting for query...",
                    interactive=False,
                    elem_classes="ret-fps-output",
                )

                query_output = gr.Textbox(
                    placeholder="Waiting for query...",
                    show_label=False,
                    interactive=False,
                )

            # Pure query input
            query_input = gr.Textbox(
                elem_id="query_input",
                placeholder="Type your query... 🚀",
                show_label=False,
                container=False,
            )
            query_input.submit(
                partial(
                    ui__query,
                    k=args.retriever.k,
                ),
                inputs=[query_input],
                outputs=[status_field, ret_fps_output, query_output],
            )

    return app


def ui(args: Box):
    app = setup_ui(args)
    app.launch()
