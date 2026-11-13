
def build_context(
    ranked_results,
    max_chunks=5,
    max_chars_per_chunk=2000
):
    """
    Build a controlled evidence context from reranked passages.
    """

    selected_chunks = []
    seen_chunks = set()
    total_chars = 0

    for result in ranked_results:

        text = result["text"].strip()
        chunk_id = result["chunk_id"]

        # Avoid duplicate chunks
        if chunk_id in seen_chunks:
            continue

        if not text:
            continue

        text = text[:max_chars_per_chunk]

        # Keep source and page information
        source = (
            f"Document: {result['document_name']}\n"
            f"Page: {result['page_number']}\n"
        )

        formatted_chunk = (
            f"{source}"
            f"Evidence:\n{text}"
        )

        if len(selected_chunks) >= max_chunks:
            break

        if total_chars + len(formatted_chunk) > 10000:
            break

        selected_chunks.append(formatted_chunk)
        seen_chunks.add(chunk_id)
        total_chars += len(formatted_chunk)

    return "\n\n---\n\n".join(selected_chunks)


def inspect_context(context):
    """Return basic context quality information."""

    return {
        "character_count": len(context),
        "word_count": len(context.split()),
        "source_count": context.count("Document:"),
        "is_empty": not bool(context.strip())
    }
