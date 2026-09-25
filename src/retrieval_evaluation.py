def evaluate_retrieval(results, all_chunks):
    """
    Evaluate retrieval results using gold supporting chunks.

    Returns:
        precision_at_k
        recall_at_k
        hit_at_k
        first_relevant_rank
    """

    # Total number of gold supporting chunks
    total_relevant = sum(
        1
        for chunk in all_chunks
        if chunk["is_supporting"]
    )

    # Number of gold chunks retrieved in Top-K
    retrieved_relevant = sum(
        1
        for result in results
        if result["is_supporting"]
    )

    k = len(results)

    # Precision@K
    if k > 0:
        precision_at_k = retrieved_relevant / k
    else:
        precision_at_k = 0.0

    # Recall@K
    if total_relevant > 0:
        recall_at_k = retrieved_relevant / total_relevant
    else:
        recall_at_k = 0.0

    # Hit@K
    hit_at_k = retrieved_relevant > 0

    # Find rank of first relevant result
    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):

        if result["is_supporting"]:
            first_relevant_rank = rank
            break

    return {
        "k": k,
        "total_relevant": total_relevant,
        "retrieved_relevant": retrieved_relevant,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_at_k": hit_at_k,
        "first_relevant_rank": first_relevant_rank
    }