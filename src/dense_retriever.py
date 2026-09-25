from sentence_transformers.util import cos_sim


def retrieve_dense(
    query,
    chunks,
    model,
    top_k=5
):
    """
    Rank chunks using dense semantic retrieval.
    """

    # Create searchable text:
    # title + chunk text
    corpus = [
        f"{chunk['title']} {chunk['text']}"
        for chunk in chunks
    ]


    # Convert query into embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )


    # Convert all chunks into embeddings
    corpus_embeddings = model.encode(
        corpus,
        convert_to_tensor=True
    )


    # Calculate cosine similarity
    scores = cos_sim(
        query_embedding,
        corpus_embeddings
    )[0]


    scored_results = []


    for chunk, score in zip(
        chunks,
        scores
    ):

        result = chunk.copy()

        result["dense_score"] = float(score)

        scored_results.append(result)


    # Highest similarity first
    scored_results.sort(
        key=lambda x: x["dense_score"],
        reverse=True
    )


    return scored_results[:top_k]