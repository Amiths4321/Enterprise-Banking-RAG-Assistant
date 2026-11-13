
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_judge import evaluate_answer
from rag_evaluation import evaluate_rag_case
from rag_test_dataset import TEST_CASES


# Mock outputs for initial pipeline validation.
# Replace these with actual RAG pipeline outputs later.

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


def run_evaluation():

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "results": [],
    }

    for test_case, output in zip(TEST_CASES, MOCK_OUTPUTS):

        print("\nEvaluating:", test_case["question"])

        retrieval_metrics = evaluate_rag_case(
            test_case,
            output,
        )

        judge_metrics = evaluate_answer(
            question=test_case["question"],
            evidence=output["evidence"],
            answer=output["answer"],
        )

        # Initial quality thresholds.
        checks = {
            "retrieval_precision": (
                retrieval_metrics["precision_at_5"] >= 0.40
            ),
            "retrieval_recall": (
                retrieval_metrics["recall_at_5"] >= 0.80
            ),
            "answer_exact_match": (
                retrieval_metrics["answer_exact_match"] == 1.0
            ),
            "citation_accuracy": (
                retrieval_metrics["citation_accuracy"] == 1.0
            ),
            "judge_faithfulness": (
                judge_metrics["faithfulness"] >= 4
            ),
            "judge_groundedness": (
                judge_metrics["groundedness"] >= 4
            ),
            "no_hallucination": (
                not judge_metrics["hallucination_detected"]
            ),
        }

        passed = all(checks.values())

        result = {
            "question": test_case["question"],
            "retrieval_metrics": retrieval_metrics,
            "judge_metrics": judge_metrics,
            "checks": checks,
            "status": "PASS" if passed else "FAIL",
        }

        report["results"].append(result)
        report["total_tests"] += 1

        if passed:
            report["passed"] += 1
        else:
            report["failed"] += 1

        print("Status:", result["status"])
        print("Checks:", checks)

    report["pass_rate"] = round(
        report["passed"] / report["total_tests"], 4
    ) if report["total_tests"] else 0

    output_file = Path("rag_evaluation_report.json")

    output_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total tests:", report["total_tests"])
    print("Passed:", report["passed"])
    print("Failed:", report["failed"])
    print("Pass rate:", report["pass_rate"] * 100, "%")
    print("Report saved:", output_file.resolve())


if __name__ == "__main__":
    run_evaluation()
