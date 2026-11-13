
import os
from ollama import Client

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://10.22.39.192:11434"
)

EMBEDDING_MODEL = "nomic-embed-text:latest"

ollama_client = Client(host=OLLAMA_HOST)


def generate_embeddings(texts, batch_size=32):
    """Generate embeddings for a list of text chunks."""

    if not texts:
        return []

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=batch
        )

        all_embeddings.extend(response["embeddings"])

    return all_embeddings


def generate_query_embedding(question):
    """Generate an embedding for a search question."""

    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    return response["embeddings"][0]
