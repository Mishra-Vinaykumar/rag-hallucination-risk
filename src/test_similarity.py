from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded successfully!")

# Question
query = (
    "Which magazine was started first "
    "Arthur's Magazine or First for Women?"
)

# Candidate sentences
sentences = [
    "Arthur's Magazine (1844–1846) was an American literary periodical.",
    
    "The magazine was started in 1989.",
    
    "The Freeway Complex Fire was a 2008 wildfire in California."
]


# Convert query into embedding
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# Convert candidate sentences into embeddings
sentence_embeddings = model.encode(
    sentences,
    convert_to_tensor=True
)


# Calculate cosine similarity
similarities = cos_sim(
    query_embedding,
    sentence_embeddings
)[0]


print("\nQuestion:")
print(query)


print("\n==============================")
print("SEMANTIC SIMILARITY RESULTS")
print("==============================")


for sentence, score in zip(
    sentences,
    similarities
):

    print("\nSentence:")
    print(sentence)

    print("Similarity:")
    print(float(score))