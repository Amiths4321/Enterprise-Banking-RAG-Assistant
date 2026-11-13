
from document_store import (
    initialise_database,
    list_documents,
    get_document_chunks
)

from vector_store import (
    get_vector_count,
    search_similar_chunks
)

from retrieval import hybrid_search


initialise_database()

print("\n========== OLD KNOWLEDGE VALIDATION ==========\n")

documents = list_documents()

print("1. DOCUMENT STORAGE")
print("Documents found:", len(documents))

all_chunks = []

for document in documents:
    chunks = get_document_chunks(document["document_id"])
    all_chunks.extend(chunks)

print("Stored chunks:", len(all_chunks))

print("\n2. VECTOR DATABASE")
print("Indexed vectors:", get_vector_count())

if not documents or not all_chunks:
    print("\nUpload and ingest banking PDFs before continuing.")
    raise SystemExit

print("\n3. SEMANTIC SEARCH")

question = "What are the loan eligibility requirements?"

semantic_results = search_similar_chunks(
    question,
    top_k=3
)

for result in semantic_results:
    metadata = result["metadata"]
    print(
        f"Document: {metadata['document_name']} | "
        f"Page: {metadata['page_number']}"
    )
    print(result["text"][:250])

print("\n4. HYBRID SEARCH")

hybrid_results = hybrid_search(
    question=question,
    chunks=all_chunks,
    top_k=3,
    candidate_k=10
)

for result in hybrid_results:
    print(
        f"Document: {result['document_name']} | "
        f"Page: {result['page_number']} | "
        f"RRF: {result['rrf_score']:.5f} | "
        f"Methods: {result['methods']}"
    )

print("\n========== VALIDATION COMPLETE ==========")
from retrieval import keyword_search, semantic_search

question = "What are the loan eligibility requirements?"

keyword_results = keyword_search(
    question,
    all_chunks,
    top_k=5
)

semantic_results = semantic_search(
    question,
    top_k=5
)

print("\nKEYWORD RESULTS:", len(keyword_results))

for result in keyword_results:
    print(result["document_name"], result["page_number"])

print("\nSEMANTIC RESULTS:", len(semantic_results))

for result in semantic_results:
    print(result["document_name"], result["page_number"])