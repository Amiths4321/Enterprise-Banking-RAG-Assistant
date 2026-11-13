
from document_store import (
    list_documents,
    get_document_chunks,
)

from document_access import get_accessible_document_ids
from rag_pipeline import answer_with_reranking


USER_ID = "local-dev"

QUESTION = (
    "What was the total income tax paid by the account holder "
    "during the 2025 financial year?"
)


def main():

    allowed_ids = get_accessible_document_ids(USER_ID)

    chunks = []

    for document in list_documents():

        document_id = document["document_id"]

        if document_id not in allowed_ids:
            continue

        chunks.extend(get_document_chunks(document_id))

    print("Authorised chunks loaded:", len(chunks))

    result = answer_with_reranking(
        question=QUESTION,
        chunks=chunks,
        top_k=5,
    )

    print("\nQUESTION:")
    print(QUESTION)

    print("\nANSWER:")
    print(result.get("answer", ""))

    print("\nSOURCES:")
    print(result.get("sources", []))


if __name__ == "__main__":
    main()
