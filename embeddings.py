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

    # Drop empty chunks early so batch positions stay meaningful
    cleaned = [t.strip() if isinstance(t, str) else "" for t in texts]
    if any(not t for t in cleaned):
        empty_positions = [i for i, t in enumerate(cleaned) if not t]
        raise ValueError(
            f"Cannot embed empty text at positions: {empty_positions[:10]}"
        )

    all_embeddings = []

    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start:start + batch_size]

        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=batch
        )

        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got {len(embeddings)} "
                f"for batch starting at index {start}."
            )

        all_embeddings.extend(embeddings)

    return all_embeddings


def generate_query_embedding(question):
    """Generate an embedding for a search question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Cannot embed an empty query (received: {question!r})."
        )

    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=question.strip()
    )

    embeddings = response["embeddings"]

    if not embeddings:
        raise RuntimeError(
            f"Embedding model returned no embeddings for query: {question!r}"
        )

    return embeddings[0]