
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vector_store import search_similar_chunks


def keyword_search(question, chunks, top_k=10):
    """Retrieve passages using TF-IDF keyword matching."""

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "document_id": chunk["document_id"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "score": float(scores[index]),
            "method": "keyword"
        })

        if len(results) >= top_k:
            break

    return results


def semantic_search(question, top_k=10):
    """Retrieve passages using ChromaDB semantic search."""

    results = search_similar_chunks(
        question=question,
        top_k=top_k
    )

    formatted_results = []

    for result in results:
        metadata = result["metadata"]

        formatted_results.append({
            "chunk_id": result["chunk_id"],
            "text": result["text"],
            "document_id": metadata["document_id"],
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "score": result["distance"],
            "method": "semantic"
        })

    return formatted_results


def reciprocal_rank_fusion(
    result_lists,
    top_k=5,
    rrf_constant=60
):
    """Combine ranked result lists using RRF."""

    scores = defaultdict(float)
    records = {}

    for results in result_lists:

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            scores[chunk_id] += (
                1.0 / (rrf_constant + rank)
            )

            if chunk_id not in records:
                records[chunk_id] = result.copy()
                records[chunk_id]["methods"] = []

            if result["method"] not in records[chunk_id]["methods"]:
                records[chunk_id]["methods"].append(
                    result["method"]
                )

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    final_results = []

    for chunk_id in ranked[:top_k]:
        result = records[chunk_id]
        result["rrf_score"] = scores[chunk_id]
        final_results.append(result)

    return final_results


def hybrid_search(question, chunks, top_k=5, candidate_k=10):
    """Combine keyword and semantic retrieval."""

    keyword_results = keyword_search(
        question,
        chunks,
        top_k=candidate_k
    )

    semantic_results = semantic_search(
        question,
        top_k=candidate_k
    )

    return reciprocal_rank_fusion(
        [keyword_results, semantic_results],
        top_k=top_k
    )
