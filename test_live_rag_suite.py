
from document_store import list_documents, get_document_chunks
from document_access import get_accessible_document_ids
from rag_pipeline import answer_with_reranking


USER_ID = "local-dev"

TEST_CASES = [
    {
        "question": "What was the closing balance in the October 2026 bank statement?",
        "expected_answer": "₹68,450.25",
        "expected_document": "PDF Bank Statement - HDFC Bank.pdf",
    },
    {
        "question": "What was the total income tax paid during the 2025 financial year?",
        "expected_answer": None,
        "expected_document": None,
    },
    {
        "question": "What was the account holder's credit score?",
        "expected_answer": None,
        "expected_document": None,
    },
]


def load_authorised_chunks():
    allowed_ids = get_accessible_document_ids(USER_ID)
    chunks = []

    for document in list_documents():
        document_id = document["document_id"]

        if document_id in allowed_ids:
            chunks.extend(get_document_chunks(document_id))

    return chunks


def run_test(case, chunks):
    result = answer_with_reranking(
        question=case["question"],
        chunks=chunks,
        top_k=5,
    )

    answer = str(result.get("answer", ""))
    sources = result.get("sources", [])

    if case["expected_answer"] is not None:
        answer_correct = case["expected_answer"] in answer

        citation_correct = any(
            source.get("document_name") == case["expected_document"]
            for source in sources
        )

        passed = answer_correct and citation_correct

    else:
        answer_lower = answer.lower()

        refusal_phrases = [
            "could not find",
            "not found",
            "not available",
            "insufficient evidence",
            "do not have enough information",
        ]

        refused = any(
            phrase in answer_lower
            for phrase in refusal_phrases
        )

        passed = refused and len(sources) == 0

    return {
        "question": case["question"],
        "passed": passed,
        "answer": answer,
        "source_count": len(sources),
    }


def main():
    chunks = load_authorised_chunks()

    print("Authorised chunks:", len(chunks))

    results = []

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\nRunning test {index}/{len(TEST_CASES)}")

        result = run_test(case, chunks)
        results.append(result)

        print("PASS" if result["passed"] else "FAIL")
        print("Sources:", result["source_count"])
        print("Answer:", result["answer"][:400])

    passed_count = sum(item["passed"] for item in results)

    print("\n========== TEST SUMMARY ==========")
    print("Total tests:", len(results))
    print("Passed:", passed_count)
    print("Failed:", len(results) - passed_count)
    print("Pass rate:", f"{passed_count / len(results) * 100:.1f}%")


if __name__ == "__main__":
    main()
