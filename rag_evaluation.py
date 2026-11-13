
import re
from typing import Any


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_ids)

    return sum(
        1 for item in retrieved if item in relevant
    ) / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]

    return sum(
        1 for item in retrieved if item in relevant
    ) / len(relevant)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def normalise_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_match_score(predicted, expected):
    return float(
        normalise_text(predicted) == normalise_text(expected)
    )


def answer_keyword_coverage(answer, evidence):
    """
    Simple lexical coverage indicator.
    This is not a semantic faithfulness metric.
    """
    answer_words = set(normalise_text(answer).split())
    evidence_words = set(normalise_text(evidence).split())

    if not answer_words:
        return 0.0

    return round(
        len(answer_words & evidence_words) / len(answer_words),
        4,
    )


def citation_accuracy(
    predicted_sources,
    expected_sources,
):
    """
    Compare cited document/page pairs with expected sources.
    """
    expected = {
        (item["document_name"], item["page_number"])
        for item in expected_sources
    }

    predicted = {
        (item["document_name"], item["page_number"])
        for item in predicted_sources
    }

    if not predicted:
        return 0.0

    return round(
        len(predicted & expected) / len(predicted),
        4,
    )


def evaluate_rag_case(test_case, system_output):

    retrieved_ids = system_output.get("retrieved_chunk_ids", [])
    answer = system_output.get("answer", "")
    evidence = system_output.get("evidence", "")
    predicted_sources = system_output.get("sources", [])

    relevant_ids = test_case.get("relevant_chunk_ids", [])
    expected_sources = test_case.get("expected_sources", [])

    expected_answer = test_case.get("expected_answer", "")

    return {
        "question": test_case["question"],
        "precision_at_5": precision_at_k(
            retrieved_ids, relevant_ids, 5
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids, relevant_ids, 5
        ),
        "mrr": mean_reciprocal_rank(
            retrieved_ids, relevant_ids
        ),
        "answer_exact_match": exact_match_score(
            answer, expected_answer
        ),
        "answer_keyword_coverage": answer_keyword_coverage(
            answer, evidence
        ),
        "citation_accuracy": citation_accuracy(
            predicted_sources, expected_sources
        ),
    }
