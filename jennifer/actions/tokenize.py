from pathlib import Path
from shutil import copy

import pandas as pd
import tiktoken

from jennifer.utilities.domains import extract_domain
from jennifer.utilities.tokenizer import split_into_many, MAX_TOKENS


def tokenize_action(url: str, rebuild: bool):
    domain = extract_domain(url)

    processed_domain_path = Path("output") / "processed" / f"{domain}.csv"

    return tokenize_local_action(processed_domain_path, rebuild)


def tokenize_local_action(input_file: Path, rebuild: bool):
    output_path = Path("output") / "processed"
    output_file = output_path / input_file.name
    tokens_path = output_path / f"{input_file.stem}-tokens.csv"
    if tokens_path.exists() and not rebuild:
        return tokens_path

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} not found, stopping early!")

    if input_file != output_file:
        copy(input_file, output_file)

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(output_file, index_col=0)
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
    return tokens_path
