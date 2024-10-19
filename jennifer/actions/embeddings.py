from pathlib import Path

import pandas as pd
from openai import OpenAI
from rich.progress import Progress


def create_embeddings_action(domain: str, tokens_path: Path, rebuild: bool):
    """
    Create embeddings for the specified domain using the tokens found at the
    provided token CSV path.
    """

    client = OpenAI()

    output_path = Path("output")
    embeddings_path = output_path / "processed" / f"{domain}-embeddings.csv"
    if embeddings_path.exists() and not rebuild:
        return embeddings_path

    if not tokens_path.exists():
        raise FileNotFoundError(f"Tokens file '{tokens_path.stem}' not found; run tokenize first!")

    # Read the tokens file, though we certainly could use the original file.
    # Maybe we could do this in parallel later, or do this at the same time
    # we're calculating the token count.
    df = pd.read_csv(tokens_path)
    with Progress() as progress:
        task = progress.add_task("Creating embeddings...", total=len(df))

        def create_embedding_local(x):
            progress.advance(task)
            return client.embeddings.create(input=x, model="text-embedding-ada-002").data[0].embedding

        # Create an embedding for each text in the dataframe, and use
        # the result to create an "embeddings" column in the output.
        df["embeddings"] = df.text.apply(create_embedding_local)

    # Write the output to a new dataframe.
    df.to_csv(embeddings_path)
    return embeddings_path
