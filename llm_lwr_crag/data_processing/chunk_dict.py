from typing import List, TypedDict

import numpy as np
import numpy.typing as npt


class ChunkDict(TypedDict):
    """
    Dictionary to be returned after the chunk embedding inside the retriever LLM.
    """

    texts: List[str]
    embeddings: npt.NDArray[np.float32]
    metadatas: List[dict]
    ids: List[str]
