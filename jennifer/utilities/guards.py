from openai import OpenAI, NotFoundError
from openai.types.fine_tuning import FineTuningJob


def ensure_success_or_abort(fine_tune_results: FineTuningJob):
    if fine_tune_results.status != "succeeded":
        print(
            f"Aborting since the job did not succeed (job {fine_tune_results.status})"
        )
        exit(1)


def retrieve_job_or_abort(client: OpenAI, job_id: str) -> FineTuningJob:
    try:
        return client.fine_tuning.jobs.retrieve(job_id)
    except NotFoundError:
        print(f"Error: Fine tuning job with job ID '{job_id}' was not found.")
        print("Run with --rebuild to create a new job.")
        exit(1)
