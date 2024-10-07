import typer

from jennifer.crawl import crawl_action
from jennifer.embeddings import create_embeddings_action
from jennifer.process_text import process_text_action
from jennifer.question import question_action
from jennifer.tokenize_action import tokenize_action

app = typer.Typer()


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
    create_embeddings_action(url, rebuild)


@app.command()
def ask_question(
    url: str, question: str, rebuild: bool = False, must_include: str = None
):
    crawl_action(url, rebuild, must_include)
    process_text_action(url, rebuild)
    tokenize_action(url, rebuild)
    create_embeddings_action(url, rebuild)
    question_action(url, question)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app()
