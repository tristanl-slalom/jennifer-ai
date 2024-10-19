from pathlib import Path

import pandas as pd
import tiktoken

from jennifer.actions.scraper.process_text import ProcessTextMetadata, process_text_metadata_from_url
from jennifer.utilities.tokenizer import split_into_many, MAX_TOKENS


class TokenizeMetadata(ProcessTextMetadata):
    tokens_path: Path


def tokenize_metadata_from_process_text(tokens_path: Path, process_text_metadata: ProcessTextMetadata):
    return TokenizeMetadata(
        tokens_path=tokens_path,
        processed_directory_path=process_text_metadata.processed_directory_path,
        processed_domain_path=process_text_metadata.processed_domain_path,
        local_domain=process_text_metadata.local_domain,
        output_path=process_text_metadata.output_path,
        text_domain_dir=process_text_metadata.text_domain_dir,
    )


def tokenize_metadata_from_url(url: str):
    process_text_metadata = process_text_metadata_from_url(url)
    tokens_path = process_text_metadata.output_path / f"{process_text_metadata.processed_domain_path.stem}-tokens.csv"
    return tokenize_metadata_from_process_text(tokens_path, process_text_metadata)


def tokenize_action(process_text_metadata: ProcessTextMetadata, rebuild: bool) -> TokenizeMetadata:
    """
    This all seems to take the input text from each row and count the number of tokens
    and splitting long lines into multiple rows.
    """
    input_file = process_text_metadata.processed_domain_path
    tokens_path = process_text_metadata.processed_directory_path / f"{input_file.stem}-tokens.csv"
    metadata = tokenize_metadata_from_process_text(tokens_path, process_text_metadata)

    if tokens_path.exists() and not rebuild:
        return metadata

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} not found, stopping early!")

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(input_file, index_col=0)
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
    return metadata
