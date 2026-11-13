
from document_store import (
    list_documents,
    get_document_chunks,
)

from document_access import get_accessible_document_ids

from retrieval import hybrid_search
from reranker import rerank_documents


def search_banking_documents(
    query: str,
    user_id: str = "local-dev",
) -> dict:

    allowed_document_ids = get_accessible_document_ids(user_id)

    if not allowed_document_ids:
        return {
            "results": [],
            "message": "No authorised banking documents available.",
        }

    all_chunks = []

    for document in list_documents():

        document_id = document["document_id"]

        if document_id not in allowed_document_ids:
            continue

        all_chunks.extend(
            get_document_chunks(document_id)
        )

    if not all_chunks:
        return {
            "results": [],
            "message": "No authorised banking documents available.",
        }

    candidates = hybrid_search(
        question=query,
        chunks=all_chunks,
        top_k=20,
        candidate_k=20,
    )

    ranked = rerank_documents(
        question=query,
        retrieved_results=candidates,
        top_k=5,
    )

    # Filter weakly relevant results.
    ranked = [
        item for item in ranked
        if item.get("rerank_score", 0) >= 0.0
    ]

    results = [
        {
            "document_name": item["document_name"],
            "page_number": item["page_number"],
            "text": item["text"],
            "rerank_score": round(item["rerank_score"], 4),
        }
        for item in ranked
    ]

    return {"results": results}


def calculate_loan_emi(
    principal: float,
    annual_interest_rate: float,
    tenure_years: float,
) -> dict:
    """
    Calculate monthly loan EMI using a fixed annual
    interest rate and monthly reducing balance.
    """

    if principal <= 0:
        raise ValueError("Principal must be greater than zero.")

    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative.")

    if tenure_years <= 0:
        raise ValueError("Tenure must be greater than zero.")

    monthly_rate = annual_interest_rate / 12 / 100
    total_months = round(tenure_years * 12)

    if monthly_rate == 0:
        emi = principal / total_months
    else:
        factor = (1 + monthly_rate) ** total_months

        emi = (
            principal
            * monthly_rate
            * factor
            / (factor - 1)
        )

    total_payment = emi * total_months
    total_interest = total_payment - principal

    return {
        "principal": round(principal, 2),
        "annual_interest_rate": annual_interest_rate,
        "tenure_years": tenure_years,
        "monthly_emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "currency": "INR",
    }


def get_document_metadata() -> dict:

    documents = list_documents()

    return {
        "document_count": len(documents),
        "documents": [
            {
                "document_id": document["document_id"],
                "document_name": document.get("document_name"),
            }
            for document in documents
        ],
    }
