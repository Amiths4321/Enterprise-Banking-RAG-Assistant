
from document_store import list_documents, get_document_chunks
from rag_pipeline import answer_with_reranking


all_chunks = []

for document in list_documents():
    chunks = get_document_chunks(document["document_id"])
    all_chunks.extend(chunks)

if not all_chunks:
    raise SystemExit("No document chunks found. Ingest PDFs first.")

question = input("Enter your banking question: ")

result = answer_with_reranking(
    question=question,
    chunks=all_chunks,
    top_k=5
)

print("\n========== AI ANSWER ==========\n")
print(result["answer"])

print("\n========== SOURCES ==========\n")

for source in result["sources"]:
    print(
        f"{source['document_name']} | "
        f"Page {source['page_number']} | "
        f"Rerank score: {source['rerank_score']:.4f}"
    )
