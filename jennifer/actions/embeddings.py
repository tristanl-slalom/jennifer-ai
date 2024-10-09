from pathlib import Path

import pandas as pd
from openai import OpenAI

from jennifer.utilities.embeddings import create_embedding


def create_embeddings_action(domain: str, tokens_path: Path, rebuild: bool):
    client = OpenAI()

    output_path = Path("output")
    embeddings_path = output_path / "processed" / f"{domain}-embeddings.csv"
    if embeddings_path.exists() and not rebuild:
        return embeddings_path

    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Tokens file '{tokens_path.stem}' not found; run tokenize first!"
        )

    df = pd.read_csv(tokens_path)
    print(f"creating embeddings for {len(df)} inputs")
    index = 0
    df_len = len(df)

    def create_embedding_local(x):
        nonlocal index
        print(f"{int(index/df_len*100)}%: {x}")
        index += 1
        return create_embedding(client, x)

    df["embeddings"] = df.text.apply(create_embedding_local)

    df.to_csv(embeddings_path)
    df.head()
    return embeddings_path
