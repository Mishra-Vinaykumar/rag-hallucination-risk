import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Convert text into simple lowercase word tokens.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def retrieve_bm25(query, chunks, top_k=5):
    """
    Rank chunks using BM25 and return the top-k results.
    """

    # Create searchable text for every chunk.
    # We include the title because the question may mention
    # the document/entity name directly.
    corpus = [
        f"{chunk['title']} {chunk['text']}"
        for chunk in chunks
    ]

    # Tokenize every chunk
    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize user question
    tokenized_query = tokenize(query)

    # Calculate score for every chunk
    scores = bm25.get_scores(tokenized_query)

    # Attach scores to chunks
    scored_results = []

    for chunk, score in zip(chunks, scores):

        result = chunk.copy()

        result["bm25_score"] = float(score)

        scored_results.append(result)

    # Highest score first
    scored_results.sort(
        key=lambda x: x["bm25_score"],
        reverse=True
    )

    # Return only top-k
    return scored_results[:top_k]