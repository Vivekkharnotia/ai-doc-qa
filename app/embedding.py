from sentence_transformers import SentenceTransformer
import numpy as np


# Load embedding model once (global level)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text_chunks: list[str]) -> np.ndarray:
    """
    Convert list of text chunks into vector embeddings.
    """
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    return embeddings
