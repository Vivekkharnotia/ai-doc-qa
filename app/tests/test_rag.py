from pdf_utils import extract_text_from_pdf
from rag_pipeline import chunk_text, create_faiss_index, search_similar_chunks
from llm import generate_answer

file_path = "../data/sample.pdf"

# Step 1: Extract text
text = extract_text_from_pdf(file_path)

# Step 2: Chunk text
chunks = chunk_text(text)

# Step 3: Create FAISS index
index, embeddings = create_faiss_index(chunks)

# Step 4: Ask question
query = "What is Vivek studying?"

retrieved_chunks = search_similar_chunks(query, index, chunks, top_k=3)

# Combine retrieved context
context = "\n\n".join(retrieved_chunks)

# Step 5: Generate final answer
answer = generate_answer(context, query)

print("\nQuestion:", query)
print("\nFinal Answer:\n")
print(answer)
