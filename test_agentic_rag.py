
from document_store import (
    list_documents,
    get_document_chunks
)

from agentic_rag import run_agentic_rag


all_chunks = []

for document in list_documents():
    chunks = get_document_chunks(
        document["document_id"]
    )
    all_chunks.extend(chunks)

if not all_chunks:
    raise SystemExit("No indexed document chunks found.")

question = input("Enter your banking question: ")

result = run_agentic_rag(
    question=question,
    chunks=all_chunks
)

print("\n========== FINAL AI ANSWER ==========\n")
print(result["answer"])

print("\n========== SEARCH HISTORY ==========\n")

for step in result["search_history"]:
    print(step)

print("\n========== SOURCES ==========\n")

for source in result["sources"]:
    print(
        f"{source['document_name']} | "
        f"Page {source['page_number']}"
    )
