from typing import List, Union

from box import Box
from handlers.auto import AutoLLM
from langchain.schema import Document

from .codeparser import CodeParser

# LLM to use for summary - cached as global variable
# to prevent multiple AutoLLM calls
llm_summary = None
llm_augment = None


def gen_summary(doc: Document, metadata_args: Box) -> None:
    """
    Generate LLM summary for the given document.
    Modifies document metadata and content in-place.
    Content is modified by prepending the summary to the original content.

    Args:
        doc (Document): Document to generate LLM summary of.
        metadata_args (Box)

    Returns:
        None
    """
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


def gen_code_structure(
    docs: Union[Document, List[Document]], metadata_args: Box
) -> None:
    """
    Attempt generating code structure for the given documents.
    In case of a single document passed, format it into a list, for easier
    implementation.
    Modifies document metadata and content in-place.
    Content is modified by prepending the concatenated list of function and
    class names.

    Args:
        docs (Union[Document, List[Document]]): Document(s) to generate code
            structure for.
        metadata_args (Box)

    Returns:
        None
    """
    if isinstance(docs, Document):
        docs = [docs]

    for doc in docs:
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
    """
    Use an LLM to augment given query, for better retrieval.

    Args:
        query (str): Query to augment with keywords and relevant file names.
        metadata_args (Box)

    Returns:
        str: Augmented query.
    """
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


def add_doc_metadata(doc: Document, metadata_args: Box) -> None:
    """
    Augment the document with pieces of metadata.
    Each metadata piece is mapped to a relevant function, used for its generation.

    Args:
        doc (Document): Document whose metadata to enrich.
        metadata_args (Box)

    Returns:
        None
    """
    if metadata_args is None:
        return

    for md_pc in metadata_args.list:
        md_gen_func = MD_PC_TO_FUNC.get(md_pc, None)
        if md_gen_func is None:
            raise ValueError(f"Invalid piece of metadata requested: {md_pc}")

        md_gen_func(doc, metadata_args)
