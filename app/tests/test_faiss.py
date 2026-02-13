from pdf_utils import extract_text_from_pdf
from rag_pipeline import chunk_text, create_faiss_index

file_path = "../data/sample.pdf"

# Extract text
text = extract_text_from_pdf(file_path)

# Chunk text
chunks = chunk_text(text)

# Create FAISS index
index, embeddings = create_faiss_index(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Total vectors stored in FAISS:", index.ntotal)
