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

    def filter_by_fp(self, ret_chunks, top_k: int = 10) -> List[str]:
        ret_files: Set[str] = set()
        for rf in ret_chunks:
            if len(ret_files) >= top_k:
                break
            ret_files.add(rf.metadata["rel_path"])

        return list(ret_files)

    @staticmethod
    def rrf(
        ret_files_1: List[str],
        ret_files_2: List[str],
        smooth_k: int = 60,
        top_k: int = 10,
    ) -> List[str]:
        scores: defaultdict = defaultdict(float)

        def rrf_make_scores(ret_files, smooth_k: int = 60):
            # Assign RRF scores in the retrieval list
            for rank, file in enumerate(ret_files, start=1):
                scores[file] += 1 / (smooth_k + rank)

        rrf_make_scores(ret_files_1, smooth_k)
        rrf_make_scores(ret_files_2, smooth_k)

        # Sort files by their total RRF scores in descending order
        # Return only the file paths in ranked order
        # Take only first k elements (i.e. top-k)
        ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [file for file, _ in ranked_files][:top_k]

    @staticmethod
    def rbf(ret_files_1, ret_files_2, p=0.8, top_k: int = 10):
        scores: defaultdict = defaultdict(float)

        def rbf_make_scores(ret_files, p):
            # Assign RBF scores in the retrieval list
            for rank, file in enumerate(ret_files, start=1):
                scores[file] += p**rank

        rbf_make_scores(ret_files_1, p)
        rbf_make_scores(ret_files_2, p)

        # Sort files by their total RRF scores in descending order
        # Return only the file paths in ranked order
        # Take only first k elements (i.e. top-k)
        ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [file for file, _ in ranked_files][:top_k]
