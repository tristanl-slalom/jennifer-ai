from pathlib import Path

import pandas as pd
from openai import AzureOpenAI
from rich.progress import Progress

from jennifer.actions.scraper.tokenize import TokenizeMetadata, tokenize_metadata_from_url


class EmbeddingsMetadata(TokenizeMetadata):
    embeddings_path: Path


def embeddings_metadata_from_tokenize(embeddings_path: Path, tokenize_metadata: TokenizeMetadata) -> EmbeddingsMetadata:
    return EmbeddingsMetadata(
        embeddings_path=embeddings_path,
        tokens_path=tokenize_metadata.tokens_path,
        processed_directory_path=tokenize_metadata.processed_directory_path,
        processed_domain_path=tokenize_metadata.processed_domain_path,
        local_domain=tokenize_metadata.local_domain,
        output_path=tokenize_metadata.output_path,
        text_domain_dir=tokenize_metadata.text_domain_dir,
    )


def embeddings_metadata_from_url(url: str) -> EmbeddingsMetadata:
    tokenize_metadata = tokenize_metadata_from_url(url)
    embeddings_path = tokenize_metadata.processed_directory_path / f"{tokenize_metadata.local_domain}-embeddings.csv"
    return embeddings_metadata_from_tokenize(embeddings_path, tokenize_metadata)


def create_embeddings_action(tokenize_metadata: TokenizeMetadata, rebuild: bool) -> EmbeddingsMetadata:
    """
    Create embeddings for the specified domain using the tokens found at the
    provided token CSV path.
    """

    client = AzureOpenAI()

    embeddings_path = tokenize_metadata.processed_directory_path / f"{tokenize_metadata.local_domain}-embeddings.csv"
    metadata = embeddings_metadata_from_tokenize(embeddings_path, tokenize_metadata)

    if embeddings_path.exists() and not rebuild:
        return metadata

    if not metadata.tokens_path.exists():
        raise FileNotFoundError(f"Tokens file '{metadata.tokens_path.stem}' not found; run tokenize first!")

    # Read the tokens file, though we certainly could use the original file.
    # Maybe we could do this in parallel later, or do this at the same time
    # we're calculating the token count.
    df = pd.read_csv(metadata.tokens_path)
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
    return metadata
