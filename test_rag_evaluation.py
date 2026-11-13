
from rag_evaluation import evaluate_rag_case
from rag_test_dataset import TEST_CASES


MOCK_OUTPUTS = [
    {
        "retrieved_chunk_ids": [
            "chunk_001",
            "chunk_004",
            "chunk_005",
        ],
        "answer": "INR 68450.25",
        "evidence": (
            "The closing balance was INR 68450.25 "
            "on the sample bank statement."
        ),
        "sources": [
            {
                "document_name": "Sample Bank Statement",
                "page_number": 1,
            }
        ],
    },
    {
        "retrieved_chunk_ids": [
            "chunk_003",
            "chunk_002",
            "chunk_006",
        ],
        "answer": "Income tax returns and bank statements",
        "evidence": (
            "Self-employed applicants must provide "
            "income tax returns and bank statements."
        ),
        "sources": [
            {
                "document_name": "Sample Loan Policy",
                "page_number": 2,
            }
        ],
    },
]


all_results = []

for test_case, output in zip(TEST_CASES, MOCK_OUTPUTS):

    result = evaluate_rag_case(test_case, output)

    all_results.append(result)

    print("\nQUESTION:", result["question"])

    for metric, value in result.items():
        if metric != "question":
            print(f"{metric}: {value}")


print("\nEVALUATION COMPLETED")

metric_names = [
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "answer_exact_match",
    "answer_keyword_coverage",
    "citation_accuracy",
]

print("\nAVERAGE METRICS")

for metric in metric_names:
    average = sum(
        result[metric] for result in all_results
    ) / len(all_results)

    print(f"{metric}: {average:.4f}")
