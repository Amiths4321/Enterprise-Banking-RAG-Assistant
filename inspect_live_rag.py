
import json
from datetime import datetime

from document_store import (
    list_documents,
    get_document_chunks,
)

from document_access import get_accessible_document_ids
from rag_pipeline import answer_with_reranking


USER_ID = "local-dev"

QUESTION = (
    "What was the closing balance in the October 2026 "
    "bank statement?"
)

EXPECTED_ANSWER = "₹68,450.25"
EXPECTED_DOCUMENT = "PDF Bank Statement - HDFC Bank.pdf"
EXPECTED_PAGE = 1


def main():

    # 1. Load authorised document chunks
    allowed_ids = get_accessible_document_ids(USER_ID)

    chunks = []

    for document in list_documents():

        document_id = document["document_id"]

        if document_id not in allowed_ids:
            continue

        chunks.extend(
            get_document_chunks(document_id)
        )

    if not chunks:
        print("No authorised document chunks found.")
        return

    print("Authorised chunks loaded:", len(chunks))
    print("Running live RAG pipeline...")

    # 2. Execute live RAG
    result = answer_with_reranking(
        question=QUESTION,
        chunks=chunks,
        top_k=5,
    )

    if not isinstance(result, dict):
        print("Unexpected result:", str(result)[:2000])
        return

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    print("\nANSWER:")
    print(str(answer)[:1500])

    # 3. Evaluate answer correctness
    answer_correct = EXPECTED_ANSWER in str(answer)

    # 4. Evaluate source precision
    relevant_sources = [
        source for source in sources
        if source.get("document_name") == EXPECTED_DOCUMENT
    ]

    source_precision = (
        len(relevant_sources) / len(sources)
        if sources else 0
    )

    # 5. Evaluate citation accuracy
    citation_correct = any(
        source.get("document_name") == EXPECTED_DOCUMENT
        and source.get("page_number") == EXPECTED_PAGE
        for source in sources
    )

    # 6. Build evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_question": QUESTION,
        "answer_correct": answer_correct,
        "source_count": len(sources),
        "relevant_source_count": len(relevant_sources),
        "source_precision": round(source_precision, 4),
        "citation_correct": citation_correct,
        "overall_status": (
            "PASS"
            if answer_correct and citation_correct
            else "FAIL"
        ),
    }

    # 7. Save report
    with open(
        "live_rag_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nLIVE RAG EVALUATION")
    print(json.dumps(report, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
