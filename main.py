from typing import Optional

import typer

from jennifer.actions.haiku import haiku_action
from jennifer.actions.vocabulary import vocabulary_action, VocabularyWord
from jennifer.actions.crawl import crawl_action, crawl_metadata_from_url
from jennifer.actions.embeddings import create_embeddings_action
from jennifer.actions.process_text import process_text_action, process_text_metadata_from_url
from jennifer.actions.question import question_action
from jennifer.actions.tokenize import tokenize_action, tokenize_metadata_from_url
from jennifer.actions.training import training_action
from jennifer.utilities.cli import argument, option

app = typer.Typer()


# This app is built on a CLI package called "Typer", which makes building
# commands for the app really easy. Below you'll see various app commands,
# most of which take a few arguments. From your virtual-environment-enabled
# terminal, once you've installed, you can execute `jennifer --help` or
# `jennifer haiku --help` to see the documentation.

# If you're using this repo for the first time, try to add a `hello-world`
# command that takes input for the name that defaults to 'world' and prints
# it to the user.

# the pattern here is, main.py defines the entry point actions and their
# options and arguments, and simply invokes an action from the
# jennifer.actions module.


@app.command()
def haiku(
    topic: option(str, "The topic of the generated haiku") = "recursion in programming",
):
    """
    Generate a haiku given the provided topic.
    """
    haiku_action(topic)


@app.command()
def crawl(
    url: argument(str, "The URL we're scraping and saving raw text files for"),
    rebuild: option(bool, "Clear the cache for the URL and rebuild it") = False,
    must_include: option(str, "Limits the scope of crawler further than the URL") = None,
):
    """
    Iterates through webpages at the provided URL and downloads them to text files.
    """
    crawl_action(url, rebuild, must_include)


@app.command()
def process_text(
    url: argument(str, "The URL to use to search for existing text files and create a text CSV"),
    rebuild: option(bool, "Clear the cache for the URL and rebuild it") = False,
):
    """
    Processes the raw data acquired from the provided URL via the crawl action and
    turns it into a single CSV.
    """
    crawl_metadata = crawl_metadata_from_url(url)
    process_text_action(crawl_metadata, rebuild)


@app.command()
def tokenize(
    url: argument(str, "The URL to use to search for existing text CSV and create a text CSV with token count"),
    rebuild: option(bool, "If set, clears all cache related to the provided URL") = False,
):
    """
    For the given URL, Processes the text CSV we got from the `process-text` action
    into a modified CSV that contains the text and number of tokens per line.
    """
    process_text_metadata = process_text_metadata_from_url(url)
    tokenize_action(process_text_metadata, rebuild)


@app.command()
def create_embeddings(
    url: argument(str, "The URL to use to look for existing tokens and create embeddings"),
    rebuild: option(bool, "If set, clears all cache related to the provided URL") = False,
):
    """
    For the given URL, Processes the CSV we got from the `tokenize` action
    into a modified CSV that contains the text, number of tokens and calculated
    embeddings.
    """

    tokenize_metadata = tokenize_metadata_from_url(url)
    create_embeddings_action(tokenize_metadata, rebuild)


@app.command()
def ask_question(
    url: argument(str, "The URL to use for all actions"),
    question: argument(str, "The question to ask using the embeddings"),
    rebuild: option(bool, "If set, clears all cache related to the provided URL") = False,
    must_include: option(str, "Limits the scope of crawler further than the URL") = None,
    max_tokens: option(int, "The maximum number of tokens for the response") = None,
):
    """
    Combine all the inputs and outputs of `crawl`, `process-text`, `tokenize`,
    and `create-embeddings` and then using the embeddings to ask a question.
    """
    crawl_data = crawl_action(url, rebuild, must_include)
    text_data = process_text_action(crawl_data, rebuild)
    tokens_data = tokenize_action(text_data, rebuild)
    embeddings_data = create_embeddings_action(tokens_data, rebuild)
    question_action(embeddings_data, question, max_tokens=max_tokens)


@app.command()
def vocabulary(
    word: argument(VocabularyWord, "The vocabulary word to use"),
    user_age: option(int, "The age of the user") = 5,
    temperature: option(Optional[float], "Sampling temperature for the generation") = None,
    top_p: option(Optional[float], "Nucleus sampling probability for the generation") = None,
):
    """
    Generate vocabulary-related responses based on the given word, user age, temperature, and top_p.
    """
    vocabulary_action(word, user_age, temperature, top_p)


@app.command()
def training(
    existing_job: option(Optional[str], "The ID of the existing training job to refer to") = None,
    test_message: option(Optional[str], "A test message to use against the trained model") = None,
    rebuild: option(bool, "If set, re-runs the training job. Expensive!") = False,
    verbose: option(bool, "If set, enables verbose output") = False,
):
    """
    Start or reuse a training job based on the given parameters.
    """
    training_action(existing_job, test_message, rebuild, verbose)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app()
