import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RAGService:

    def __init__(self):

        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # LLM for answer generation
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        self.index = None
        self.chunks = []

    def create_index(self, text):

        # Split text into chunks
        self.chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        embeddings = self.embed_model.encode(self.chunks)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings))

        # Create storage folder
        os.makedirs("app/storage", exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, "app/storage/faiss.index")

        # Save text chunks
        np.save("app/storage/chunks.npy", self.chunks)

    def answer_question(self, question):

        if self.index is None:
            return "No document uploaded."

        question_embedding = self.embed_model.encode([question])

        distances, indices = self.index.search(
            np.array(question_embedding), k=3
        )

        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        context = " ".join(retrieved_chunks)

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.generator.generate(
            **inputs,
            max_length=150
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer