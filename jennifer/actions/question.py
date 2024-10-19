from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from jennifer.utilities.context import create_context


def question_action(embeddings_path: Path, question: str, max_tokens=None, stop_sequence=None):
    client = OpenAI()

    df = pd.read_csv(embeddings_path, index_col=0)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

    df.head()

    context = create_context(client, question, df, max_len=100000)

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
