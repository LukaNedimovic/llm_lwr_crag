from abc import ABC
from collections import defaultdict
from typing import List, Set, Tuple

from langchain.schema import Document


class AbstractDB(ABC):
    """
    Abstract handler class for seamless integration with various databases.
    """

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Store embeddings in the Chroma database.
        """
        pass

    def query(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the Chroma database for files.
        This modified version returns both the file paths and their similarity scores.
        """
        return []

    @staticmethod
    def filter_by_fp(
        all_ret_chunks: List[Document], top_k: int = 10
    ) -> Tuple[List[str], List[Document]]:
        """
        Filter returned chunks by file paths. Take at most K unique file paths.

        Args:
            all_ret_chunks (List[Document]): List of all retrieved chunks.
            top_k (int): Return at most `top_k` unique file paths.

        Returns:
            Tuple[List[str], List[Document]]: A tuple consisting of:
                (1) ret_fps (List[str]): List of unique, retrieved file paths
                (2) ret_chunks (List[Document]): List of accompanying chunks.
        """
        ret_chunks: List[str] = []
        ret_fps: Set[str] = set()
        for rc in all_ret_chunks:
            if len(ret_fps) >= top_k:
                break
            ret_chunks.append(rc)
            ret_fps.add(rc.metadata["rel_path"])

        return list(ret_fps), ret_chunks

    @staticmethod
    def rrf(
        ret_fps_1: List[str],
        ret_fps_2: List[str],
        smooth_k: int = 60,
        top_k: int = 10,
    ) -> List[str]:
        """
        Implement standard Reciprocal Rank Fusion (RRF).
        Read more about it herE: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

        Args:
            ret_fps_1 (List[str]): First list of retrieved file paths.
            ret_fps_2 (List[str]): Second list of retrieved file paths.
            smooth_k (int): Smoothing factor.
            top_k (int): Take at most `top_k` files.

        Returns:
            List[str]: Top-K list, merged from the `ret_fps_1` and `ret_fps_2`.
        """
        scores: defaultdict = defaultdict(float)

        def rrf_make_scores(ret_files, smooth_k: int = 60):
            # Assign RRF scores in the retrieval list
            for rank, file in enumerate(ret_files, start=1):
                scores[file] += 1 / (smooth_k + rank)

        rrf_make_scores(ret_fps_1, smooth_k)
        rrf_make_scores(ret_fps_2, smooth_k)

        # Sort files by their total RRF scores in descending order
        # Return only the file paths in ranked order
        # Take only first k elements (i.e. top-k)
        ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [file for file, _ in ranked_files][:top_k]

    @staticmethod
    def rbf(
        ret_fps_1: List[str], ret_fps_2: List[str], p: float = 0.8, top_k: int = 10
    ):
        """
        Implement Reciprocal Rank-Based Fusion.
        Formula: sum(p ** rank)

        Args:
            ret_fps_1 (List[str]): First list of retrieved file paths.
            ret_fps_2 (List[str]): Second list of retrieved file paths.
            p (float): Smoothing factor.
            top_k (int): Take at most `top_k` files.

        Returns:
            List[str]: Top-K list, merged from the `ret_fps_1` and `ret_fps_2`.
        """
        scores: defaultdict = defaultdict(float)

        def rbf_make_scores(ret_files, p):
            # Assign RBF scores in the retrieval list
            for rank, file in enumerate(ret_files, start=1):
                scores[file] += p**rank

        rbf_make_scores(ret_fps_1, p)
        rbf_make_scores(ret_fps_2, p)

        # Sort files by their total RRF scores in descending order
        # Return only the file paths in ranked order
        # Take only first k elements (i.e. top-k)
        ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [file for file, _ in ranked_files][:top_k]

    @staticmethod
    def fetch_by_fp(ret_fps: List[str], chunks: List[Document]) -> List[Document]:
        """
        Fetch the chunks from the given file paths.

        Args:
            ret_fps (List[str]): List of retrieved file paths.
            chunks (List[Documeent]): List of relevant chunks.

        Returns:
            ret_chunks (List[Document]): List of chunks matched from their
                file paths.
        """
        ret_chunks: List[Document] = []
        for fp in ret_fps:
            chunks_with_fp = []
            # TODO: Make this more pythonic
            for ch in chunks:
                if ch.metadata["rel_path"] == fp:
                    chunks_with_fp.append(ch)
            ret_chunks.append(chunks_with_fp[0])  # Append chunk with given fp
        return ret_chunks
