from app.index import query
from app.scoring import cosine_distance_to_similarity


def search_documents(question: str) -> list[dict]:
    """Search the knowledge base."""
    results = query(question, top_k=5)

    return [
        {
            "text": document,
            "doc_name": metadata["doc_name"],
            "chunk_index": metadata["chunk_index"],
            "similarity": cosine_distance_to_similarity(distance),
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
