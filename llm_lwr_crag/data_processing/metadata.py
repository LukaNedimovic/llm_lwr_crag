from typing import List, Union

from box import Box
from handlers.auto import AutoLLM
from langchain.schema import Document

from .codeparser import CodeParser

# LLM to use for summary - cached as global variable
# to prevent multiple AutoLLM calls
llm_summary = None


def gen_summaries(
    documents: Union[Document, list[Document]], metadata_args: Box
) -> None:
    # Check whether it is the first time instantiating the LLM for summary
    # If so, instantiate it and keep it cached
    global llm_summary
    if llm_summary is None:
        llm_summary_args = metadata_args.llm_summary
        llm_summary = AutoLLM.from_args(llm_summary_args)

    if isinstance(documents, Document):
        documents.metadata["llm_summary"] = llm_summary.gen_summary(documents)
        documents.page_content = (
            f"LLM Summary: {documents.metadata['llm_summary']}"
            "\n\n"
            f"Content: {documents.page_content}"
        )
        return

    # Generate summaries
    llm_summary.gen_summaries(documents)

    # Move summary to content
    for doc in documents:
        doc.page_content = (
            f"LLM Summary: {doc.metadata['llm_summary']}"
            "\n\n"
            f"Content: {doc.page_content}"
        )
        # doc.metadata.pop("llm_summary", None)


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


def empty(*args):
    pass


# Metadata piece to function mapping
MD_PC_TO_FUNC = {
    "llm_summary": gen_summaries,
    "code_structure": gen_code_structure,
}


def add_document_metadata(doc: Document, metadata_args):
    for md_pc in metadata_args.list:
        md_gen_func = MD_PC_TO_FUNC.get(md_pc, None)
        if md_gen_func is None:
            raise ValueError(f"Invalid piece of metadata requested: {md_pc}")

        md_gen_func(doc, metadata_args)


def add_documents_metadata(documents: List[Document], metadata_args):
    for md_pc in metadata_args.list:
        md_gen_func = MD_PC_TO_FUNC.get(md_pc, None)
        if md_gen_func is None:
            raise ValueError(f"Invalid piece of metadata requested: {md_pc}")

        md_gen_func(documents, metadata_args)
