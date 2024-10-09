from typing import List

from openai import OpenAI
from scipy import spatial


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def create_embedding(client: OpenAI, input: str):
    return (
        client.embeddings.create(input=input, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )
