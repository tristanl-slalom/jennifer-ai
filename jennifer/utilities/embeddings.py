from typing import List

from scipy import spatial


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """
    Originally this was a method in OpenAI's utilities module, but they seemed
    to have removed it. I found this online; it seems to do the trick.

    So far we only seem to use cosine, and this function could be simplified to
    just do that, but I keep the original version I found here just in case.
    Basically, seems to take a list of floats and a bunch of lists of floats and
    calculates the cosine of each entry between them.

    Math stuff, I dunno. Fancy!
    """
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
