from bm25_retriever import retrieve_bm25
from dense_retriever import retrieve_dense


def min_max_normalize(values):
    """
    Normalize values into the range 0 to 1.
    """

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def retrieve_hybrid(
    query,
    chunks,
    model,
    top_k=5,
    alpha=0.5
):
    """
    Hybrid retrieval using normalized BM25
    and Dense similarity scores.

    alpha controls Dense weight.

    alpha = 0.5 means:
    50% Dense + 50% BM25
    """

    # Get scores for ALL chunks
    bm25_results = retrieve_bm25(
        query=query,
        chunks=chunks,
        top_k=len(chunks)
    )

    dense_results = retrieve_dense(
        query=query,
        chunks=chunks,
        model=model,
        top_k=len(chunks)
    )


    # --------------------------------
    # Map scores using chunk IDs
    # --------------------------------

    bm25_score_map = {
        result["chunk_id"]: result["bm25_score"]
        for result in bm25_results
    }

    dense_score_map = {
        result["chunk_id"]: result["dense_score"]
        for result in dense_results
    }


    # Keep original chunk order
    bm25_scores = [
        bm25_score_map[chunk["chunk_id"]]
        for chunk in chunks
    ]

    dense_scores = [
        dense_score_map[chunk["chunk_id"]]
        for chunk in chunks
    ]


    # --------------------------------
    # Normalize scores
    # --------------------------------

    normalized_bm25 = min_max_normalize(
        bm25_scores
    )

    normalized_dense = min_max_normalize(
        dense_scores
    )


    # --------------------------------
    # Combine scores
    # --------------------------------

    hybrid_results = []

    for (
        chunk,
        bm25_score,
        dense_score,
        norm_bm25,
        norm_dense
    ) in zip(
        chunks,
        bm25_scores,
        dense_scores,
        normalized_bm25,
        normalized_dense
    ):

        hybrid_score = (
            alpha * norm_dense
            +
            (1 - alpha) * norm_bm25
        )

        result = chunk.copy()

        result["bm25_score"] = bm25_score
        result["dense_score"] = dense_score

        result["normalized_bm25"] = norm_bm25
        result["normalized_dense"] = norm_dense

        result["hybrid_score"] = hybrid_score

        hybrid_results.append(result)


    # Highest hybrid score first
    hybrid_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    return hybrid_results[:top_k]