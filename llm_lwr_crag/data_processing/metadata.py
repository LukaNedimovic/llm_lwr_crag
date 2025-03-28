from typing import Union

from box import Box
from handlers.auto import AutoLLM
from langchain.schema import Document

from .codeparser import CodeParser

# LLM to use for summary - cached as global variable
# to prevent multiple AutoLLM calls
llm_summary = None
llm_augment = None


def gen_summary(doc: Document, metadata_args: Box) -> None:
    # Check whether it is the first time instantiating the LLM for summary
    # If so, instantiate it and keep it cached
    global llm_summary
    if llm_summary is None:
        llm_summary_args = metadata_args.llm_summary
        llm_summary = AutoLLM.from_args(llm_summary_args)

    # Add summary to metadata and move it to content
    doc.metadata["llm_summary"] = llm_summary.gen_summary(doc)
    doc.page_content = (
        f"LLM Summary: {doc.metadata['llm_summary']}"
        "\n\n"
        f"Content: {doc.page_content}"
    )


def gen_code_structure(documents: Union[Document, list[Document]], metadata_args: Box):
    if isinstance(documents, Document):
        documents = [documents]

    for doc in documents:
        functions, classes = CodeParser.parse_code(
            doc.page_content,
            doc.metadata["ext"],
        )
        doc.metadata["functions"] = ", ".join([fp for f in functions for fp in f])
        doc.metadata["classes"] = ", ".join([fclp for cl in classes for fclp in cl])

        doc.page_content = (
            f"Functions: {doc.metadata['functions']}"
            "\n\n"
            f"Classes: {doc.metadata['classes']}"
            "\n\n"
            f"Content: {doc.page_content}"
        )


def augment_query(query: str, metadata_args: Box) -> str:
    # Check whether it is the first time instantiating the LLM for summary
    # If so, instantiate it and keep it cached
    global llm_augment
    if llm_augment is None:
        llm_augment_args = metadata_args.augment_query
        llm_augment = AutoLLM.from_args(llm_augment_args)

    # Augment the query
    return llm_augment.augment(query)


# Metadata piece to function mapping
MD_PC_TO_FUNC = {
    "code_structure": gen_code_structure,
    "llm_summary": gen_summary,
}


def add_doc_metadata(doc: Document, metadata_args: Box):
    for md_pc in metadata_args.list:
        md_gen_func = MD_PC_TO_FUNC.get(md_pc, None)
        if md_gen_func is None:
            raise ValueError(f"Invalid piece of metadata requested: {md_pc}")

        md_gen_func(doc, metadata_args)
