from typing import Optional

import typer

from jennifer.actions.embeddings import embeddings_action
from jennifer.actions.haiku import haiku_action
from jennifer.actions.vocabulary import vocabulary_action, VocabularyWord
from jennifer.actions.scraper.crawl import crawl_action
from jennifer.actions.scraper.embeddings import create_embeddings_action
from jennifer.actions.scraper.process_text import process_text_action
from jennifer.actions.question import question_action
from jennifer.actions.scraper.tokenize import tokenize_action
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
def ask_question(
    url: argument(str, "The URL to use for all actions"),
    question: argument(str, "The question to ask using the embeddings"),
    rebuild: option(bool, "If set, clears all cache related to the provided URL") = False,
    must_include: option(str, "Limits the scope of crawler further than the URL") = None,
    max_tokens: option(int, "The maximum number of tokens for the response") = None,
):
    """
    Crawls and scrapes a given URL, processes the data into embeddings and asks a question
    of the embedding data.
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
    existing_job: option(Optional[str], "A training job ID, defaulting to a previous job run if available") = None,
    test_message: option(Optional[str], "A test message to use against the trained model") = None,
    rebuild: option(bool, "If set, re-runs the training job even if one is remembered. Expensive!") = False,
    verbose: option(bool, "If set, enables verbose output") = False,
):
    """
    Start or reuse a 'baseball vs hockey' training job based on the given parameters and tests it
    with a message.
    """
    training_action(existing_job, test_message, rebuild, verbose)


embeddings_app = typer.Typer()
app.add_typer(embeddings_app, name="embeddings", help="Various sub-commands around generating and using embeddings")


@embeddings_app.command("build")
def embeddings_build(
    rebuild: option(bool, "If set, re-runs the embeddings process even if output exists locally") = True,
    visualization: option(bool, "If set, shows a visualization of the embeddings in 2D") = False,
    classification: option(bool, "If set, shows a visualization of the our ability to classify ratings") = False,
):
    """
    Leveraging the Amazon Fine Food Reviews, generate some embeddings and get recommendations. Will default
    to rebuilding embeddings, even if they exist.
    """
    embeddings_action(
        rebuild,
        visualization,
        classification,
        False,
        None,
        None,
        None,
        None,
        0.0,
    )


@embeddings_app.command()
def search(
    search_term: argument(str, "Searches for products matching the description"),
    rebuild: option(bool, "If set, re-runs the embeddings process even if output exists locally") = False,
    visualization: option(bool, "If set, shows a visualization of the embeddings in 2D") = False,
    classification: option(bool, "If set, shows a visualization of the our ability to classify ratings") = False,
    num_results: option(int, "How many results to return if product description is set") = 1,
):
    """
    Search Amazon Food Reviews with embeddings for a specific search term. Will default to reusing
    embeddings if they exist.
    """
    embeddings_action(
        rebuild,
        visualization,
        classification,
        False,
        search_term,
        num_results,
        None,
        None,
        0.0,
    )


@embeddings_app.command()
def clusters(
    rebuild: option(bool, "If set, re-runs the embeddings process even if output exists locally") = False,
    visualization: option(bool, "If set, shows a visualization of the embeddings in 2D") = False,
    classification: option(bool, "If set, shows a visualization of the our ability to classify ratings") = False,
    show_clustering: option(bool, "If set, shows a visualization of the review clustering") = False,
    num_clusters: option(int, "How many clusters to identify in the data") = 4,
    num_reviews_per_cluster: option(int, "How many reviews to show for each cluster") = 3,
    min_similarity: option(
        float, "When calculating clusters that match a product, what's the minimum similarity"
    ) = 0.3,

):
    """
    Determine clusters of common themes across the reviews. Will default to reusing
    embeddings if they exist.
    """
    embeddings_action(
        rebuild,
        visualization,
        classification,
        show_clustering,
        None,
        None,
        num_clusters,
        num_reviews_per_cluster,
        min_similarity,
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app()
