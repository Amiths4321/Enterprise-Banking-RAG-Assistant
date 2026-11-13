
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker_model = CrossEncoder(MODEL_NAME)


def rerank_documents(question, retrieved_results, top_k=5):
    """
    Rerank retrieved passages using a cross-encoder.

    Each input result must contain:
    - chunk_id
    - text
    - document_name
    - page_number
    """

    if not retrieved_results:
        return []

    pairs = [
        [question, result["text"]]
        for result in retrieved_results
    ]

    scores = reranker_model.predict(pairs)

    ranked_results = []

    for result, score in zip(retrieved_results, scores):
        updated_result = result.copy()

        updated_result["rerank_score"] = float(score)

        ranked_results.append(updated_result)

    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]
