from embedding import get_embeddings

texts = [
    "Machine learning is amazing.",
    "I love building AI systems."
]

embeddings = get_embeddings(texts)

print("Shape of embeddings:", embeddings.shape)
print("First vector (first 5 values):", embeddings[0][:5])
