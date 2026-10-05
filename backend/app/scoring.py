def cosine_distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to the app's rounded similarity score."""
    return round(1 - distance, 2)
