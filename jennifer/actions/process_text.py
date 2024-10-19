from pathlib import Path

import pandas as pd

from jennifer.actions.crawl import CrawlMetadata, crawl_metadata_from_url
from jennifer.utilities.text import remove_newlines


class ProcessTextMetadata(CrawlMetadata):
    processed_directory_path: Path
    processed_domain_path: Path


def process_text_metadata_from_crawl(
    processed_directory_path: Path, processed_domain_path: Path, crawl_metadata: CrawlMetadata
):
    return ProcessTextMetadata(
        processed_directory_path=processed_directory_path,
        processed_domain_path=processed_domain_path,
        local_domain=crawl_metadata.local_domain,
        output_path=crawl_metadata.output_path,
        text_domain_dir=crawl_metadata.text_domain_dir,
    )


def process_text_metadata_from_url(url: str):
    crawl_metadata = crawl_metadata_from_url(url)
    processed_directory_path = crawl_metadata.output_path / "processed"
    processed_domain_path = processed_directory_path / f"{crawl_metadata.local_domain}.csv"
    return ProcessTextMetadata.from_crawl(processed_directory_path, processed_domain_path, crawl_metadata)


def process_text_action(crawl_metadata: CrawlMetadata, rebuild: bool) -> ProcessTextMetadata:
    processed_directory_path = crawl_metadata.output_path / "processed"
    processed_domain_path = processed_directory_path / f"{crawl_metadata.local_domain}.csv"
    metadata = process_text_metadata_from_crawl(processed_directory_path, processed_domain_path, crawl_metadata)

    if processed_domain_path.exists() and not rebuild:
        return metadata

    processed_directory_path.mkdir(parents=True, exist_ok=True)

    # Create a list to store the text files
    texts = []

    # Get all the text files in the text directory
    for file in crawl_metadata.text_domain_dir.iterdir():
        # Open the file and read the text
        with open(file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append(
                (
                    file.name[len(crawl_metadata.local_domain) + 2 :]
                    .replace("index.html", "")
                    .replace(".txt", "")
                    .replace("-", " ")
                    .replace("_", " ")
                    .replace("#update", ""),
                    text,
                )
            )

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=["fname", "text"])

    # Set the text column to be the raw text with the newlines removed
    df["text"] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(processed_domain_path)
    return metadata
