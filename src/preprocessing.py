def extract_documents(record):
    """
    Convert one HotpotQA record into structured document records.
    """

    query_id = record["id"]

    context = record["context"]

    titles = context["title"]
    sentences = context["sentences"]

    # Titles containing the correct supporting evidence
    supporting_titles = set(
        record["supporting_facts"]["title"]
    )

    documents = []

    for index, (title, sentence_list) in enumerate(
        zip(titles, sentences),
        start=1
    ):

        # Combine the sentence list into one document
        document_text = " ".join(
            sentence.strip()
            for sentence in sentence_list
        )

        document = {
            "query_id": query_id,
            "document_id": f"D{index:03d}",
            "title": title,
            "text": document_text,
            "is_supporting": title in supporting_titles
        }

        documents.append(document)

    return documents

def create_chunks(record):
    """
    Convert one HotpotQA record into sentence-level chunks.
    Mark only exact HotpotQA supporting sentences as supporting.
    """

    query_id = record["id"]

    context = record["context"]

    titles = context["title"]
    sentences = context["sentences"]

    # Create exact (title, sentence_id) pairs
    supporting_pairs = set(
        zip(
            record["supporting_facts"]["title"],
            record["supporting_facts"]["sent_id"]
        )
    )

    chunks = []

    for document_index, (title, sentence_list) in enumerate(
        zip(titles, sentences),
        start=1
    ):

        document_id = f"D{document_index:03d}"

        for sentence_index, sentence in enumerate(sentence_list):

            clean_sentence = sentence.strip()

            chunk = {
                "query_id": query_id,
                "document_id": document_id,

                # +1 only for human-friendly chunk numbering
                "chunk_id": f"{document_id}_C{sentence_index + 1:03d}",

                "title": title,
                "text": clean_sentence,

                # Exact supporting fact check
                "is_supporting": (
                    title,
                    sentence_index
                ) in supporting_pairs
            }

            chunks.append(chunk)

    return chunks