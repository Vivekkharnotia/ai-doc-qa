import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List


class RAGService:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.base_storage_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "storage"
        )

        os.makedirs(self.base_storage_path, exist_ok=True)

    def _get_doc_path(self, document_id: str):
        return os.path.join(self.base_storage_path, document_id)

    def _get_index_path(self, document_id: str):
        return os.path.join(self._get_doc_path(document_id), "faiss.index")

    def _get_chunks_path(self, document_id: str):
        return os.path.join(self._get_doc_path(document_id), "chunks.npy")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def create_index(self, text: str, document_id: str):
        chunks = self.chunk_text(text)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Create document directory
        doc_path = self._get_doc_path(document_id)
        os.makedirs(doc_path, exist_ok=True)

        # Save index
        faiss.write_index(index, self._get_index_path(document_id))

        # Save chunks
        np.save(self._get_chunks_path(document_id), np.array(chunks, dtype=object))

        print(f">>> SAVED DOCUMENT: {document_id}")

        return len(chunks)

    def search(self, query: str, document_id: str, top_k: int = 3) -> List[str]:
        index_path = self._get_index_path(document_id)
        chunks_path = self._get_chunks_path(document_id)

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise ValueError("Document not found. Upload first.")

        # Load index and chunks
        index = faiss.read_index(index_path)
        chunks = np.load(chunks_path, allow_pickle=True).tolist()

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            results.append(chunks[i])

        return results
