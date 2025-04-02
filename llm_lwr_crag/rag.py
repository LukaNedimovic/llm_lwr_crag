from typing import List, Set, Tuple, Union

from handlers import AbstractDB, AbstractLLM
from langchain.schema import Document


class Retriever:
    def __init__(
        self,
        ret_db_vec: AbstractDB,
        ret_db_bm25: AbstractDB,
        ret_rerank: AbstractLLM,
    ):
        self.db_vec = ret_db_vec
        self.bm25 = ret_db_bm25
        self.rerank = ret_rerank

    def __call__(self, query: str, k: int = 10) -> Tuple[List[str], List[Document]]:
        # Query the database to get top-K relevant files
        # Corrective factor of 4 used as an example - can be changed
        # as it is a hyperparameter
        ret_chunks = self.db_vec.query(query, k=4 * k)
        ret_fps = None

        # Apply BM25, if applicable
        # Hybrid search with BM25 is done based on file paths,
        # therefore file paths must be filtered out first,
        # and then the chunks
        if self.bm25:
            ret_chunks_bm25 = self.bm25.query(query, k=k // 2)
            ret_fps_bm25, ret_chunks_bm25 = AbstractDB.filter_by_fp(
                ret_chunks_bm25, top_k=k // 2
            )

            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)
            ret_fps = AbstractDB.rbf(ret_fps, ret_fps_bm25)
            ret_chunks = AbstractDB.fetch_by_fp(ret_fps, ret_chunks + ret_chunks_bm25)

        # Apply reranking, if applicable
        # Reranking is done based on the file content
        # Filter out the files after reranking them
        if self.rerank:
            ret_chunks = self.rerank.rerank(query, ret_chunks)
            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)

        # In case over pure vector search, make sure to filter out file paths
        if ret_fps is None:
            ret_fps, ret_chunks = AbstractDB.filter_by_fp(ret_chunks, top_k=k)

        return ret_fps, ret_chunks


class Generator:
    def __init__(self, gen_llm: AbstractLLM):
        self.llm = gen_llm

    def __call__(self, query: str, ret_chunks: List[Document]) -> Union[str, None]:
        return self.llm.generate(query, ret_chunks) if self.llm else None


class RAG:
    def __init__(
        self,
        ret_db_vec: AbstractDB,
        ret_db_bm25: AbstractDB,
        ret_rerank: AbstractLLM,
        gen_llm: AbstractLLM,
    ):
        self.retriever = Retriever(ret_db_vec, ret_db_bm25, ret_rerank)
        self.generator = Generator(gen_llm)

    def __call__(self, query: str, k: int = 10):
        # Retrieve top K chunks
        ret_fps, ret_chunks = self.retriever(query, k)
        # Generate an answer based on retrieved chunks
        gen_ans = self.generator(query, ret_chunks)

        return ret_fps, ret_chunks, gen_ans

    @staticmethod
    def recall(
        ret_fps: List[str], ground_truth_fps: Set[str]
    ) -> Tuple[float, Set[str]]:
        ret_relevant = ground_truth_fps.intersection(ret_fps)

        # Calculate Recall@K and add it to the total_recall for future calculation
        recall = len(ret_relevant) / len(ground_truth_fps) if ground_truth_fps else 0.0
        return recall, ret_relevant
