from pathlib import Path

import pandas as pd
import tiktoken

from jennifer.utilities.domains import extract_domain
from jennifer.utilities.tokenizer import split_into_many, MAX_TOKENS


def tokenize_action(url: str, rebuild: bool):
    domain = extract_domain(url)

    output_path = Path("output")
    processed_domain_path = output_path / "processed" / f"{domain}.csv"
    tokens_path = output_path / "processed" / f"{domain}-tokens.csv"
    if tokens_path.exists() and not rebuild:
        return

    if not processed_domain_path.exists():
        raise FileNotFoundError(
            f"processed domain path for {domain} not found; run process first!"
        )

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(processed_domain_path, index_col=0)
    df.columns = ["title", "text"]

    # Tokenize the text and save the number of tokens to a new column
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]["text"] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]["n_tokens"] > MAX_TOKENS:
            shortened += split_into_many(tokenizer, row[1]["text"])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]["text"])

    df = pd.DataFrame(shortened, columns=["text"])
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    df.to_csv(tokens_path)
