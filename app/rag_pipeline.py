import faiss
import numpy as np
from embedding import get_embeddings


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_faiss_index(chunks: list[str]):
    """
    Creates a FAISS index from text chunks.
    Returns the index and the embeddings.
    """
    embeddings = get_embeddings(chunks)

    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


def search_similar_chunks(query: str, index, chunks: list[str], top_k: int = 3):
    """
    Searches for the most similar chunks to the query.
    """
    # Convert query into embedding
    query_embedding = get_embeddings([query])

    # Perform similarity search
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve matching chunks
    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results
