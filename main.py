from random import choice

from openai import OpenAI
from pathlib import Path

import typer

from jennifer.crawl import crawl_action
from jennifer.embeddings import create_embeddings_action
from jennifer.process_text import process_text_action
from jennifer.question import question_action
from jennifer.tokenize_action import tokenize_action

app = typer.Typer()


@app.command()
def tng_episode(title: str, output_path: Path = Path('./episode.mp3')):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Come up with a short premise for a hypothetical new Star Trek "
                f"the next generation episode that fits the title '{title}'. It should be "
                "a maximum of 3 sentences."
            }
        ],
    )

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    synopsis = completion.choices[0].message.content
    chosen_voice = choice(voices)
    print(f"{chosen_voice}: {synopsis}")
    speech_file_path = output_path
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=chosen_voice,
        input=synopsis
    ) as response:
        response.stream_to_file(speech_file_path)


@app.command()
def hello_ai():
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write an essay about the importance of brushing your teeth."
            }
        ],
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


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
def ask_question(url: str, question: str, rebuild: bool = False, must_include: str = None):
    crawl_action(url, rebuild, must_include)
    process_text_action(url, rebuild)
    tokenize_action(url, rebuild)
    create_embeddings_action(url, rebuild)
    question_action(url, question)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app()
