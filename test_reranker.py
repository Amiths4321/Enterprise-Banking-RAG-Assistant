
from reranker import rerank_documents

question = (
    "What are the eligibility requirements "
    "for a self-employed personal loan applicant?"
)

sample_results = [
    {
        "chunk_id": "chunk_1",
        "text": "The bank offers competitive personal loan interest rates.",
        "document_name": "Personal Loan Policy",
        "page_number": 1
    },
    {
        "chunk_id": "chunk_2",
        "text": "Self-employed applicants must submit income tax returns "
                "and bank statements for the previous two years.",
        "document_name": "Personal Loan Policy",
        "page_number": 4
    },
    {
        "chunk_id": "chunk_3",
        "text": "Salaried applicants must provide salary slips and "
                "employment proof.",
        "document_name": "Personal Loan Policy",
        "page_number": 3
    },
    {
        "chunk_id": "chunk_4",
        "text": "The applicant must meet the minimum age and "
                "credit score requirements specified by the bank.",
        "document_name": "Personal Loan Policy",
        "page_number": 2
    }
]

results = rerank_documents(
    question,
    sample_results,
    top_k=4
)

print("\nRERANKED RESULTS\n")

for rank, result in enumerate(results, start=1):
    print(f"Rank: {rank}")
    print(f"Document: {result['document_name']}")
    print(f"Page: {result['page_number']}")
    print(f"Score: {result['rerank_score']:.4f}")
    print(f"Text: {result['text']}")
    print("-" * 50)
