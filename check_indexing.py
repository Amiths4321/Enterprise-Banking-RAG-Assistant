
from document_store import list_documents, get_document_chunks
from vector_store import get_vectors_by_document, get_vector_count

documents = list_documents()

print("Total vectors:", get_vector_count())

for document in documents:
    document_id = document["document_id"]

    sqlite_chunks = get_document_chunks(document_id)
    chroma_records = get_vectors_by_document(document_id)

    print("\nDocument:", document["document_name"])
    print("SQLite chunks:", len(sqlite_chunks))
    print("Chroma vectors:", len(chroma_records["ids"]))
