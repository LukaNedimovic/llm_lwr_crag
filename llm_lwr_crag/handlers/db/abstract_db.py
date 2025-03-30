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
    def filter_by_fp(all_ret_chunks, top_k: int = 10) -> Tuple[List[str], List[str]]:
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
    def rbf(ret_fps_1: List[str], ret_fps_2: List[str], p=0.8, top_k: int = 10):
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
    def fetch_by_fp(ret_fps: List[str], chunks: List[Document]):
        ret_chunks: List[Document] = []
        for fp in ret_fps:
            chunks_with_fp = []
            # TODO: Make this more pythonic
            for ch in chunks:
                if ch.metadata["rel_path"] == fp:
                    chunks_with_fp.append(ch)
            ret_chunks.append(chunks_with_fp[0])  # Append chunk with given fp
        return ret_chunks
