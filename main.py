from pathlib import Path
from typing import Optional

import typer

from jennifer.actions.haiku import haiku_action
from jennifer.actions.vocabulary import vocabulary_action, VocabularyWord
from jennifer.actions.crawl import crawl_action
from jennifer.actions.embeddings import create_embeddings_action
from jennifer.actions.process_text import process_text_action
from jennifer.actions.question import question_action
from jennifer.actions.tokenize import tokenize_action, tokenize_local_action
from jennifer.actions.training import training_action
from jennifer.utilities.domains import extract_domain

app = typer.Typer()


@app.command()
def haiku(topic="recursion in programming"):
    haiku_action(topic)


@app.command()
def crawl(url: str, rebuild: bool = False, must_include: str = None):
    crawl_action(url, rebuild, must_include)


@app.command()
def process_text(url: str, rebuild: bool = False):
    process_text_action(url, rebuild)


@app.command()
def tokenize(url: str, rebuild: bool = False):
    tokenize_action(url, rebuild)


@app.command()
def create_embeddings(url: str, rebuild: bool = False):
    domain = extract_domain(url)
    tokens_path = Path("output") / "processed" / f"{domain}-tokens.csv"
    create_embeddings_action(domain, tokens_path, rebuild)


@app.command()
def ask_question(
    url: str,
    question: str,
    rebuild: bool = False,
    must_include: str = None,
    max_tokens: int = None,
):
    domain = extract_domain(url)

    crawl_action(url, rebuild, must_include)
    process_text_action(url, rebuild)
    tokens_path = tokenize_action(url, rebuild)
    embeddings_path = create_embeddings_action(domain, tokens_path, rebuild)
    question_action(embeddings_path, question, max_tokens=max_tokens)


@app.command()
def ask_question_local(input_file: Path, question: str, rebuild: bool = False):
    tokens_path = tokenize_local_action(input_file, rebuild)
    embeddings_path = create_embeddings_action(input_file.stem, tokens_path, rebuild)
    question_action(embeddings_path, question)


@app.command()
def vocabulary(
    word: VocabularyWord,
    user_age: int = 5,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    vocabulary_action(word, user_age, temperature, top_p)


@app.command()
def training(existing_job: str = None, test_message: str = None):
    training_action(existing_job, test_message)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app()
