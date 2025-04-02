from pathlib import Path

import pandas as pd
import tiktoken
from openai import AzureOpenAI
from pandas import DataFrame
from rich.progress import Progress

from jennifer.utilities.embeddings import create_embedding


def load_review_data(input_file_path: Path, max_tokens: int):
    df = pd.read_csv(input_file_path, index_col=0)
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()  # Return a new Series with missing values removed
    df["combined"] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    # subsample to 1k most recent reviews and remove samples that are too long
    top_n = 1000
    # first cut to first 2k entries, assuming less than half will be filtered out
    df = df.sort_values("Time").tail(top_n * 2)
    df.drop("Time", axis=1, inplace=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    # omit reviews that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    return df


def add_embedding_column(client: AzureOpenAI, df: DataFrame) -> DataFrame:
    with Progress() as progress:
        task = progress.add_task("Creating embeddings...", total=len(df))

        def create_embedding_local(x):
            progress.advance(task)
            return create_embedding(client, x)

        # Create an embedding for each text in the dataframe, and use the result to create
        # an "embedding_string" column in the output. Note: confusingly, embeddings_string
        # isn't a string at this point, it's actually an array. When it gets saved to/loaded
        # from a CSV later though, it'll be a string.
        df["embeddings_string"] = df.combined.apply(lambda x: create_embedding_local(x))

    return df


def safely_write_to_csv(df: DataFrame, output_file_path: Path):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file_path)
