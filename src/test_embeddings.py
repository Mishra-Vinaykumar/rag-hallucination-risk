from sentence_transformers import SentenceTransformer


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded successfully!")


sentence = "Arthur's Magazine was published in the 19th century."

embedding = model.encode(sentence)


print("\nOriginal sentence:")
print(sentence)

print("\nEmbedding shape:")
print(embedding.shape)

print("\nFirst 10 numbers of embedding:")
print(embedding[:10])