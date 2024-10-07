from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from jennifer.utilities.context import create_context
from jennifer.utilities.domains import extract_domain


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
                    "content": "Answer the question based on the context below, and if the question "
                               "can't be answered based on the context, say \"I don't know\"\n\n",
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
