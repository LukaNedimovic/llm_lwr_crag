from typing import List, Set, Tuple, Union

import pandas as pd
import pipeline as pl
from box import Box
from handlers import AbstractDB, AbstractLLM
from langchain.schema import Document
from utils.logging import log_tc


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
        """
        Retrieve top-K files for the given query.

        Args:
            query (str): Query for which to retrieve relevant file paths / chunks.
            k (int): Retrieve top-k files.

        Returns:
            Tuple[List[str], List[Document]]: List of top-K file paths (fps)
                and chunks.
        """
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
        """
        Generate the answer for the given query, based on the retrieved chunks.

        Args:
            query (str): Query to generate answer off of.
            ret_chunks (List[Document]): List of retrieved chunks.

        Returns:
            Union[str, None]: If generator is defined, return the textual answer.
                If not, return None.
        """
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

    def __call__(
        self, query: str, k: int = 10
    ) -> Tuple[List[str], List[Document], Union[str, None]]:
        """
        Perform a single retrieval + generation task.

        Args:
            query (str): Query for which to retrieve relevant file paths / chunks.
            k (int): Retrieve top-k files.

        Returns:
            Tuple[List[str], List[Document], str]: A tuple consisting of:
                (1) ret_fps (List[str]): retrieved file paths
                (2) ret_chunks (List[Document]): retrieved chunks
                (3) gen_ans (Union[str, None]): Generated, textual answer to the query.
                    If not defined, will return None.
        """
        # Retrieve top K chunks
        ret_fps, ret_chunks = self.retriever(query, k)
        # Generate an answer based on retrieved chunks
        gen_ans = self.generator(query, ret_chunks)

        return ret_fps, ret_chunks, gen_ans

    @staticmethod
    def recall(
        ret_fps: List[str], ground_truth_fps: Set[str]
    ) -> Tuple[float, Set[str]]:
        """
        Implement a standard recall metric.

        Args:
            ret_fps (List[str]): List of retrieved, uniqe file paths
            ground_truth_fps (Set[str]): Set of ground truth files,
                from evaluation dataset.

        Returns:
            Tuple[float, Set[str]]: A tuple consisting of:
                (1) recall (float)
                (2) ret_relevant (Set[str]): Intersection of ground truth files
                    and retrieved files.
        """
        ret_relevant = ground_truth_fps.intersection(ret_fps)

        # Calculate Recall@K and add it to the total_recall for future calculation
        recall = len(ret_relevant) / len(ground_truth_fps) if ground_truth_fps else 0.0
        return recall, ret_relevant

    @staticmethod
    def from_args(args: Box, docs: List[Document], chunks: List[Document]):
        """
        Initialize `RAG` object given arguments, and documents / chunks.

        Args:
            args (Box): A wrapped-up parsed YAML arguments.
            docs (List[Document])
            chunks (List[Document])

        Returns:
            RAG
        """
        ret_db_vec, ret_db_bm25, ret_rerank = pl.setup_retrieval(args, docs, chunks)
        gen_llm = pl.setup_generation(args)
        return RAG(ret_db_vec, ret_db_bm25, ret_rerank, gen_llm)

    def eval(self, eval_df: pd.DataFrame, k: int = 10) -> float:
        """
        Evaluate the system with Recall@K metric.

        Args:
            ret_db_vec (AbstractDBHandler): Database to query.
            eval_df (pd.DataFrame): Evaluation data.
            k (int = 10): Top-K files to be retrieved

        Returns:
            avg_recall (float): Average Recall@K (defaults to Recall@10) over the
            evaluation dataset.
        """
        total_recall = 0.0

        for tc_id, row in eval_df.iterrows():
            query = row["question"]
            ground_truth_fps = set(row["files"])

            # Retrieve relevant file paths and chunks
            ret_fps, ret_chunks, gen_ans = self(query, k)

            # Calculate Recall@K for the query
            recall, ret_relevant = RAG.recall(ret_fps, ground_truth_fps)
            total_recall += recall

            # Log the test case
            log_tc(
                tc_id=tc_id,
                num_tc=len(eval_df),
                query=query,
                ground_truth_fps=ground_truth_fps,
                ret_fps=ret_fps,
                ret_relevant=ret_relevant,
                recall=recall,
                gen_ans=gen_ans,
            )

        # Average Recall@K across all questions
        avg_recall = total_recall / len(eval_df)
        return avg_recall
