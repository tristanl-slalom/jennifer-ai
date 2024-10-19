from datetime import datetime, timezone

from openai.types.fine_tuning import FineTuningJob
from pandas import DataFrame
from sklearn.utils import Bunch


def print_model_id(fine_tune_results: FineTuningJob):
    ft_model = fine_tune_results.fine_tuned_model
    print(f"Fine-tuned Model ID: '{ft_model}'")


def print_fine_tuning_results(results: DataFrame, verbose: bool):
    if not verbose:
        return

    last_result = results[results["train_accuracy"].notnull()].tail(1)
    print(f"{last_result}")


def print_sports_statistics(sports_dataset: Bunch, verbose: bool):
    if not verbose:
        return

    len_all, len_baseball, len_hockey = (
        len(sports_dataset.data),
        len([e for e in sports_dataset.target if e == 0]),
        len([e for e in sports_dataset.target if e == 1]),
    )
    print(f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}")


def print_job_outcome(fine_tune_results: FineTuningJob, job_id: str):
    dt_utc = datetime.fromtimestamp(fine_tune_results.finished_at, tz=timezone.utc)
    dt_local = dt_utc.astimezone()
    time_string = dt_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"Job '{job_id}' {fine_tune_results.status} at {time_string}")
