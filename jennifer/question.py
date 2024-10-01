from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from openai import OpenAI

from scipy import spatial

from jennifer.utilities import extract_domain


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


def question_action(url: str, question: str, max_tokens=150, stop_sequence=None):
    client = OpenAI()

    domain = extract_domain(url)
    output_path = Path("output")
    df = pd.read_csv(
        output_path / "processed" / f"{domain}-embeddings.csv", index_col=0
    )
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

    df.head()

    context = create_context(client, question, df)

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n",
                },
                {
                    "role": "user",
                    f"content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(e)
        return ""


def create_context(client: OpenAI, question, df, max_len=1800):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = (
        client.embeddings.create(input=question, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)
