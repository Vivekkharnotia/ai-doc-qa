from pdf_utils import extract_text_from_pdf
from rag_pipeline import chunk_text, create_faiss_index, search_similar_chunks

file_path = "../data/sample.pdf"

# Step 1: Extract text
text = extract_text_from_pdf(file_path)

# Step 2: Chunk text
chunks = chunk_text(text)

# Step 3: Create FAISS index
index, embeddings = create_faiss_index(chunks)

# Step 4: Ask a question
query = "What is Vivek studying?"

results = search_similar_chunks(query, index, chunks, top_k=2)

print("\nQuery:", query)
print("\nTop Retrieved Chunks:\n")

for i, chunk in enumerate(results):
    print(f"Result {i+1}:\n")
    print(chunk)
    print("-" * 50)
