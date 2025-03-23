from typing import List

from box import Box
from handlers.auto import AutoLLM
from langchain.schema import Document

# LLM to use for summary - cached as global variable
# to prevent multiple AutoLLM calls
llm_summary = None


def gen_summaries(documents: list[Document], metadata_args: Box) -> None:
    # Check whether it is the first time instantiating the LLM for summary
    # If so, instantiate it and keep it cached
    global llm_summary
    if llm_summary is None:
        llm_summary_args = metadata_args.llm_summary
        llm_summary = AutoLLM.from_args(llm_summary_args)

    # Generate summaries
    llm_summary.gen_summaries(documents)

    # Move summary to content
    for document in documents:
        document.page_content = (
            f"LLM Summary: {document.metadata['llm_summary']}"
            "\n\n"
            "Content: {document.page_content}"
        )
        document.metadata.pop("llm_summary", None)


# Metadata piece to function mapping
MD_PC_TO_FUNC = {
    "llm_summary": gen_summaries,
}


def add_document_metadata(documents: List[Document], metadata_args):
    for md_pc in metadata_args.list:
        md_gen_func = MD_PC_TO_FUNC.get(md_pc, None)
        if md_gen_func is None:
            raise ValueError(f"Invalid piece of metadata requested: {md_pc}")

        md_gen_func(documents, metadata_args)
