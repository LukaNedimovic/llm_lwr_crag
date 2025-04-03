from typing import List, Tuple

import faiss
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from utils.logging import logger

from .abstract_db import AbstractDB


class FAISSHandler(AbstractDB):
    def __init__(self, args):
        self.collection_name = args.collection_name

        index = faiss.IndexFlatL2(len(args.emb_func.embed_query("init")))
        self.db = FAISS(
            embedding_function=args.emb_func,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def __str__(self):
        return "FAISS"

    def add_documents(self, chunks: List[Document]) -> None:
        logger.info(f"Adding embeddings into the {self.collection_name} (FAISS)...")
        self.db.add_documents(chunks)
        logger.info("Successfully added embeddings into the FAISS database!")

    def query(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        ret_chunks = self.db.similarity_search(query, k=k)
        return ret_chunks
