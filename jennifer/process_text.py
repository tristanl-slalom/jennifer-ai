from pathlib import Path

import pandas as pd


def _remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def process_text_action(domain: str, rebuild: bool):
    domain = domain[8:domain.index("/", 8)]
    text_domain_path = Path("text") / domain
    processed_directory_path = Path("processed")
    processed_domain_path = processed_directory_path / f"{domain}.csv"

    if processed_domain_path.exists() and not rebuild:
        return

    processed_directory_path.mkdir(parents=True, exist_ok=True)

    # Create a list to store the text files
    texts = []

    # Get all the text files in the text directory
    for file in text_domain_path.iterdir():

        # Open the file and read the text
        with open(file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append(
                (
                    file.name[len(domain)+2:].replace('index.html', '')
                    .replace('.txt', '')
                    .replace('-', ' ')
                    .replace('_', ' ')
                    .replace('#update', ''),
                    text
                )
            )

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + _remove_newlines(df.text)
    df.to_csv(f"processed/{domain}.csv")
    df.head()
