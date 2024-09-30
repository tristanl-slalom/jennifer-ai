import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


def _create_embedding(client: OpenAI, input: str):
    print(f"creating embedding: {input}")
    return client.embeddings.create(input=input, model='text-embedding-ada-002').data[0].embedding


def create_embeddings_action(domain: str, rebuild: bool):
    client = OpenAI()

    domain = domain[8:domain.index("/", 8)]

    embeddings_path = Path(f'processed/{domain}-embeddings.csv')
    if embeddings_path.exists() and not rebuild:
        return

    df = pd.read_csv(f'processed/{domain}-tokens.csv')
    print(f"creating embeddings for {len(df)} inputs")
    df['embeddings'] = df.text.apply(
        lambda x: _create_embedding(client, x)
    )

    df.to_csv(embeddings_path)
    df.head()
