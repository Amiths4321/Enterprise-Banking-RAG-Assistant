
import chromadb
from pathlib import Path

from embeddings import (
    generate_embeddings,
    generate_query_embedding
)


DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "banking_documents"

chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def index_document_chunks(chunks):
    """Generate embeddings and persist document chunks."""

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page_number": int(chunk["page_number"]),
                "chunk_index": int(chunk["chunk_index"])
            }
            for chunk in chunks
        ]
    )

    return len(chunks)


def search_similar_chunks(question, top_k=5, document_id=None):
    """Find relevant chunks using semantic similarity."""

    query_embedding = generate_query_embedding(question)

    search_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }

    if document_id:
        search_arguments["where"] = {
            "document_id": document_id
        }

    results = collection.query(**search_arguments)

    matches = []

    if not results["ids"]:
        return matches

    for i, chunk_id in enumerate(results["ids"][0]):
        matches.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return matches


def get_vector_count():
    return collection.count()
