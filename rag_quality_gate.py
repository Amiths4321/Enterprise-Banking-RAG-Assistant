
def validate_rag_response(result, min_sources=1):
    """
    Basic evidence and citation validation.
    """

    if not isinstance(result, dict):
        return False, "Invalid response format."

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    if not isinstance(answer, str) or not answer.strip():
        return False, "Empty answer."

    if not isinstance(sources, list):
        return False, "Invalid source format."

    if len(sources) < min_sources:
        return False, "Insufficient supporting sources."

    valid_sources = [
        source for source in sources
        if isinstance(source, dict)
        and source.get("document_name")
        and source.get("page_number") is not None
    ]

    if not valid_sources:
        return False, "No valid document citations."

    return True, "Evidence and citation checks passed."
