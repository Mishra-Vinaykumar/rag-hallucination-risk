import pandas as pd


def save_retrieval_trace(
    record,
    results,
    output_path,
    retrieval_method="BM25"
):
    """
    Save retrieval results for one query to a CSV file.
    """

    trace_rows = []

    for rank, result in enumerate(results, start=1):

        row = {
            "query_id": record["id"],
            "question": record["question"],
            "expected_answer": record["answer"],
            "retrieval_method": retrieval_method,
            "rank": rank,
            "document_id": result["document_id"],
            "chunk_id": result["chunk_id"],
            "title": result["title"],
            "text": result["text"],
            "retrieval_score": result["bm25_score"],
            "is_supporting": result["is_supporting"]
        }

        trace_rows.append(row)

    dataframe = pd.DataFrame(trace_rows)

    dataframe.to_csv(
        output_path,
        index=False
    )

    return dataframe