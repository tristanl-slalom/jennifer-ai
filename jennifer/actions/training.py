import time
from pathlib import Path
from typing import List, Optional
import base64

import pandas as pd
from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob
from pandas import DataFrame
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch

from jennifer.utilities.guards import ensure_success_or_abort, retrieve_job_or_abort
from jennifer.utilities.printery import (
    print_model_id,
    print_fine_tuning_results,
    print_sports_statistics,
    print_job_outcome,
)

SAMPLE_BASEBALL_TWEET = """
BREAKING: The Tampa Bay Rays are finalizing a deal to acquire slugger Nelson Cruz 
from the Minnesota Twins, sources tell ESPN.
"""

SAMPLE_HOCKEY_TWEET = """
Thank you to the @Canes and all you amazing Caniacs that have been so supportive!
You guys are some of the best fans in the NHL without a doubt! Really excited to
start this new chapter in my career with the @DetroitRedWings!!
"""

CONTEXT_MESSAGE = {
    "role": "system",
    "content": "This is a functional chatbot that figures out if a given message "
               "is related to baseball or hockey. Only print 'hockey', 'baseball' or "
               "'not'",
}


class TrainingMessage(BaseModel):
    role: str
    content: str


class TrainingRow(BaseModel):
    messages: List[TrainingMessage]


class TrainingMetadataPaths(BaseModel):
    metadata: Path
    output: Path
    results: Path


class TrainingMetadata(BaseModel):
    job_id: str
    paths: TrainingMetadataPaths


def training_action(
    existing_job: Optional[str],
    test_message: Optional[str],
    rebuild: bool,
    verbose: bool,
):
    """
    Week 4 course on model training from the jupyter notebook at
    https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb
    and based on the fine-tuning guide available at
    https://platform.openai.com/docs/guides/fine-tuning
    """

    # Gather the sports dataset and explore it.
    sports_dataset = _explore_data(verbose)

    # Prepare the training data for our model.
    paths = _prepare_data(sports_dataset)

    # Fine-tune the base model with our training data.
    client = OpenAI()
    fine_tune_results = _fine_tune(client, existing_job, paths, rebuild, verbose)

    # Use the fine-tuned model from the job with a test string
    _use_model(client, fine_tune_results.model, test_message)


def _explore_data(verbose: bool) -> Bunch:
    """
    The fetch_20newsgroups function in the sklearn datasets package is used to load the
    20 newsgroups dataset, which is a collection of approximately 20,000 newsgroup documents,
    partitioned across 20 different newsgroups. This dataset is commonly used for experimenting
    with text classification and clustering algorithms.
    """

    categories = ["rec.sport.baseball", "rec.sport.hockey"]
    sports_dataset = fetch_20newsgroups(
        subset="train", shuffle=True, random_state=42, categories=categories
    )

    print_sports_statistics(sports_dataset, verbose)
    return sports_dataset


def _prepare_data(sports_dataset: Bunch) -> TrainingMetadataPaths:
    """
    Generate a training dataset for the open AI gpt-4o-mini model from the data
    we explored earlier. This model demands a set of messages rather than a prompt
    and completion like previous models (i.e. babbage or ada).

    We use this to establish:
    - that the system is operating like a function returning either 'baseball'
    or 'hockey'.
    - that the user provides a message from the dataset
    - that the assistant returns either 'baseball' or 'hockey' for that message

    :return: a TrainingMetadataPaths for various file paths for the training job.
    """

    context_message = TrainingMessage(
        role=CONTEXT_MESSAGE["role"], content=CONTEXT_MESSAGE["content"]
    )
    # get all the labels from the source dataset
    labels = [
        sports_dataset.target_names[x].split(".")[-1] for x in sports_dataset["target"]
    ]
    # get all the content text from the source dataset
    texts = [text.strip() for text in sports_dataset["data"]]

    # The goal is to generate a "jsonl" file, which contains one json object per line.
    # I'm using Pydantic to lay out these rows in the format expected by the model during
    # training.
    training_rows = [
        TrainingRow(
            messages=[
                context_message,
                TrainingMessage(role="user", content=text),
                TrainingMessage(role="assistant", content=label),
            ]
        )
        for text, label in zip(texts, labels)
    ]

    # The various paths supporting this job.
    training_output_path = Path("output") / "training"
    data_output_path = training_output_path / "sports-data.jsonl"
    metadata_path = training_output_path / f"{data_output_path.stem}-meta.json"
    result_path = training_output_path / f"{data_output_path.stem}-results.csv"
    training_output_path.mkdir(parents=True, exist_ok=True)

    # Write the training rows one JSON line at a time to the output path.
    with open(data_output_path, "w") as f:
        for training_row in training_rows:
            json_line = training_row.json()
            f.write(f"{json_line}\n")

    return TrainingMetadataPaths(
        metadata=metadata_path,
        output=data_output_path,
        results=result_path,
    )


def _fine_tune(
    client: OpenAI,
    existing_job: str,
    paths: TrainingMetadataPaths,
    rebuild: bool,
    verbose: bool,
) -> FineTuningJob:
    """
    Using the training data we prepared in the previous step, kick off a
    training job for the base model. If we have an existing job we've
    kicked off previously, use (or wait for) that instead, unless rebuild
    is True.
    """
    job_id = _create_or_reuse_tuning_job(client, paths, existing_job, rebuild)
    fine_tune_results = retrieve_job_or_abort(client, job_id)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Fine-Tuning Job '{job_id}': {fine_tune_results.status}"
        )

        while fine_tune_results.status not in ["succeeded", "failed", "cancelled"]:
            # While the job isn't in a terminal state, let's wait and occasionally check in.
            # the job may take upwards of 40 minutes, so let's ping once every 30 seconds.
            job_description = f"Fine-Tuning Job '{job_id}': {fine_tune_results.status}"
            progress.update(task, description=job_description)
            time.sleep(30)
            fine_tune_results = client.fine_tuning.jobs.retrieve(job_id)

    # Once the job has reached its final status, print what happened to it.
    print_job_outcome(fine_tune_results, job_id)

    # Stop here if the job wasn't successful.
    ensure_success_or_abort(fine_tune_results)

    # Examine how our job did.
    results = gather_fine_tune_results(client, fine_tune_results, paths)
    print_fine_tuning_results(results, verbose)

    # Show off our swanky new model.
    print_model_id(fine_tune_results)

    return fine_tune_results


def _use_model(client: OpenAI, model: str, test_message: Optional[str]):
    """
    Given the model from the fine-tuning job, run the test message against it.
    """

    def baseball_or_hockey(message: str) -> str:
        """
        A quick function to call our fine-tuned model, looking for its one-word classification
        """
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                CONTEXT_MESSAGE,
                {"role": "user", "content": message + "\n\n###\n\n"},
            ],
            max_tokens=10,
            temperature=0,
        )
        return chat_completion.choices[0].message.content

    if not test_message:
        # We have a couple of samples that should work.
        result = baseball_or_hockey(SAMPLE_HOCKEY_TWEET)
        print(f"Sample hockey tweet determined to be '{result}' related")
        result = baseball_or_hockey(SAMPLE_BASEBALL_TWEET)
        print(f"Sample baseball tweet determined to be '{result}' related")
    else:
        # Use the user-provided test message.
        result = baseball_or_hockey(test_message)
        print(f"Provided test message determined to be {result} related")


def gather_fine_tune_results(
    client: OpenAI, fine_tune_results: FineTuningJob, paths: TrainingMetadataPaths
) -> DataFrame:
    """
    Fine-tuning jobs have result files we can pull and examine. I don't understand
    their contents yet though.
    :return:
    """
    fine_tune_result_files = fine_tune_results.result_files
    result_file = client.files.retrieve(fine_tune_result_files[0])
    content = client.files.content(result_file.id)
    # save content to file
    with open(paths.results, "wb") as f:
        f.write(base64.b64decode(content.text.encode("utf-8")))
    results = pd.read_csv(paths.results)
    return results


def _create_or_reuse_tuning_job(
    client: OpenAI,
    paths: TrainingMetadataPaths,
    existing_job: Optional[str],
    rebuild: bool,
):
    """
    Whether by creation, user-specification, or loading from previous jobs, get a job ID.
    Saves the job ID and associated paths for future use. My finest work (not really).
    :return: The job ID of the training job for our data.
    """

    if existing_job:
        _update_metadata(existing_job, paths)
        return existing_job

    # If we have a metadata file that contains a job ID, let's presume we want to use that.
    if paths.metadata.exists() and not rebuild:
        job_id = _get_job_id_from_saved_metadata(paths)
        _update_metadata(job_id, paths)
        return job_id

    # At this point we've exhausted all avenues where we don't have to make a new job.
    # Time to make a new one!
    if not rebuild:
        print(f"No job ID from a previous fine-tuning job found. Creating a new one!")
    elif paths.metadata.exists():
        print(f"Discarding previous fine-tuning job to create a new one")

    train_file = client.files.create(file=open(paths.output, "rb"), purpose="fine-tune")
    fine_tuning_job = client.fine_tuning.jobs.create(
        training_file=train_file.id, model="gpt-4o-mini-2024-07-18"
    )
    print(
        f"Job '{fine_tuning_job.id}' created! This may take 30-40 minutes to complete..."
    )
    job_id = fine_tuning_job.id
    _update_metadata(job_id, paths)
    return job_id


def _get_job_id_from_saved_metadata(paths: TrainingMetadataPaths):
    with open(paths.metadata, "r") as f:
        data = f.read()
        metadata: TrainingMetadata = TrainingMetadata.model_validate_json(data)
        job_id = metadata.job_id
        print(
            f"Using fine-tuning job ID '{job_id}' from a previous run. Use --rebuild to create a new model."
        )
    return job_id


def _update_metadata(job_id: str, paths: TrainingMetadataPaths):
    """Write the job ID into the metadata file for next time."""

    metadata = TrainingMetadata(job_id=job_id, paths=paths)
    with open(paths.metadata, "w") as f:
        f.write(metadata.json())
