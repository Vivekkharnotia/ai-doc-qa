from pdf_utils import extract_text_from_pdf
from rag_pipeline import chunk_text

file_path = "../data/sample.pdf"

text = extract_text_from_pdf(file_path)
chunks = chunk_text(text)

print("Total text length:", len(text))
print("Number of chunks:", len(chunks))
print("\nFirst chunk preview:\n")
print(chunks[0])
