import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import base64

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.datasets import fetch_20newsgroups


class TrainingMessage(BaseModel):
    role: str
    content: str


class TrainingRow(BaseModel):
    messages: List[TrainingMessage]


def training_action(existing_job: Optional[str], test_message: Optional[str]):
    """
    Week 4 course on model training from the jupyter notebook at
    https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb
    """
    client = OpenAI()

    categories = ["rec.sport.baseball", "rec.sport.hockey"]
    sports_dataset = fetch_20newsgroups(
        subset="train", shuffle=True, random_state=42, categories=categories
    )

    print(sports_dataset["data"][0])
    print(sports_dataset.target_names[sports_dataset["target"][0]])
    len_all, len_baseball, len_hockey = (
        len(sports_dataset.data),
        len([e for e in sports_dataset.target if e == 0]),
        len([e for e in sports_dataset.target if e == 1]),
    )
    print(
        f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}"
    )

    context_message_dict = {
        "role": "system",
        "content": "Marv is a functional chatbot that figures out if a given message is related to baseball or hockey",
    }
    context_message: TrainingMessage = TrainingMessage.model_validate(context_message_dict)

    labels = [sports_dataset.target_names[x].split('.')[-1] for x in sports_dataset['target']]
    texts = [text.strip() for text in sports_dataset['data']]

    rows = [
        TrainingRow(messages=[
            context_message,
            TrainingMessage(role="user", content=t),
            TrainingMessage(role="assistant", content=l)
        ])
        for t, l in zip(texts, labels)
    ]

    training_path = Path("output") / "training"
    output_path = training_path / "sport2.jsonl"
    result_path = training_path / f"{output_path.stem}-results.csv"
    training_path.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for r in rows:
            json_line = r.json()
            f.write(f"{json_line}\n")

    train_file = client.files.create(file=open(output_path, "rb"), purpose="fine-tune")
    if existing_job is None:
        fine_tuning_job = client.fine_tuning.jobs.create(
          training_file=train_file.id,
          model="gpt-4o-mini-2024-07-18"
        )
        job_id = fine_tuning_job.id
    else:
        job_id = existing_job

    fine_tune_results = client.fine_tuning.jobs.retrieve(job_id)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Fine-Tuning Job '{job_id}': {fine_tune_results.status}")

        while fine_tune_results.status not in ["succeeded", "failed", "cancelled"]:
            progress.update(task, description=f"Fine-Tuning Job '{job_id}': {fine_tune_results.status}")
            time.sleep(60)
            fine_tune_results = client.fine_tuning.jobs.retrieve(job_id)

        # Convert the Unix timestamp to a datetime object in UTC
        dt_utc = datetime.fromtimestamp(fine_tune_results.finished_at, tz=timezone.utc)

        # Convert the datetime object to the local timezone
        dt_local = dt_utc.astimezone()

        # Format the datetime object as a string
        time_string = dt_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')

        print(f"Job '{job_id}' {fine_tune_results.status} at {time_string}")

        if fine_tune_results.status == "succeeded":
            fine_tune_result_files = client.fine_tuning.jobs.retrieve(job_id).result_files
            result_file = client.files.retrieve(fine_tune_result_files[0])
            content = client.files.content(result_file.id)
            # save content to file
            with open(result_path, "wb") as f:
                f.write(base64.b64decode(content.text.encode("utf-8")))

            results = pd.read_csv(result_path)
            last_result = results[results['train_accuracy'].notnull()].tail(1)
            print(f"{last_result}")
        else:
            print("Aborting since the job did not complete successfully")
            exit(1)

        test = pd.read_json(output_path, lines=True)
        print(test.head())

        ft_model = fine_tune_results.fine_tuned_model

        # note that this calls the legacy completions api - https://platform.openai.com/docs/api-reference/completions
        test_baseball_content = {"role": "user", "content": test['messages'][0][1]['content'] + '\n\n###\n\n'}
        res = client.chat.completions.create(
            model=ft_model,
            messages=[context_message_dict, test_baseball_content],
            max_tokens=10,
            temperature=0
        )
        print(res.choices[0].message.content)

        res = client.chat.completions.create(
            model=ft_model,
            messages=[context_message_dict, test_baseball_content],
            max_tokens=10,
            temperature=0,
            logprobs=True,
        )
        print(res.choices[0].logprobs.content[0].logprob)

        if not test_message:
            sample_hockey_tweet = """
            Thank you to the @Canes and all you amazing Caniacs that have been so supportive!
            You guys are some of the best fans in the NHL without a doubt! Really excited to
            start this new chapter in my career with the @DetroitRedWings!!
            """
            res = client.chat.completions.create(
                model=ft_model,
                messages=[context_message_dict, {"role": "user", "content": sample_hockey_tweet + '\n\n###\n\n'}],
                max_tokens=10,
                temperature=0,
            )
            print(f"Sample hockey tweet determined to be {res.choices[0].message.content} related")

            sample_baseball_tweet = """
            BREAKING: The Tampa Bay Rays are finalizing a deal to acquire slugger Nelson Cruz 
            from the Minnesota Twins, sources tell ESPN.
            """
            res = client.chat.completions.create(
                model=ft_model,
                messages=[context_message_dict, {"role": "user", "content": sample_baseball_tweet + '\n\n###\n\n'}],
                max_tokens=10,
                temperature=0,
            )
            print(f"Sample baseball tweet determined to be {res.choices[0].message.content} related")
        else:
            res = client.chat.completions.create(
                model=ft_model,
                messages=[context_message_dict, {"role": "user", "content": test_message + '\n\n###\n\n'}],
                max_tokens=10,
                temperature=0,
            )
            print(f"Provided test message determined to be {res.choices[0].message.content} related")
