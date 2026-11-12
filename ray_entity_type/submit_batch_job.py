"""
Submit a job to AWS Batch from a YAML config (copied from mle-common/models/genre/automation).
"""

from datetime import datetime
from dataclasses import dataclass
from time import sleep
from typing import Any

import boto3
import click
import yaml
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class BatchParams:
    """
    Data structure to hold batch job parameters.
    """

    queue: str
    definition: str
    timeout_seconds: int


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_batch_params(cfg: dict[str, Any]) -> BatchParams:
    batch = cfg["batch_params"]
    return BatchParams(
        queue=batch["batch_job_queue"],
        definition=batch["batch_job_definition"],
        timeout_seconds=int(batch["batch_job_timeout"]),
    )


def build_environment(config: dict[str, dict[str, Any]], env_var_prefix: str) -> list[dict[str, str]]:
    env_vars: list[dict[str, str]] = []
    sections_to_process = ["input_params", "output_params"]

    for section in sections_to_process:
        if section not in config:
            continue
        for key, value in config[section].items():
            env_vars.append({"name": f"{env_var_prefix}_{key.upper()}", "value": str(value)})
    return env_vars


def log_environment_vars(environment: list[dict[str, str]]) -> None:
    logger.info("Environment variables:")
    for item in sorted(environment, key=lambda x: x["name"]):
        logger.info("env %s=%s", item["name"], item["value"])


def submit_batch_job(
    client: Any,
    params: BatchParams,
    job_name: str,
    environment: list[dict[str, str]],
) -> str:
    logger.info(
        "Submitting Batch job %s to queue %s with definition %s and timeout of %s seconds...",
        job_name,
        params.queue,
        params.definition,
        params.timeout_seconds,
    )
    response = client.submit_job(
        jobName=job_name,
        jobQueue=params.queue,
        jobDefinition=params.definition,
        containerOverrides={"environment": environment},
        propagateTags=True,
        timeout={"attemptDurationSeconds": params.timeout_seconds},
    )
    return response["jobId"]


def observe_batch_job(job_id: str) -> None:
    client = boto3.client("batch")
    while True:
        job = client.describe_jobs(jobs=[job_id])["jobs"][0]
        job_status = job["status"]
        logger.info(f"Job {job_id} status: {job_status}")
        if job_status in ("SUCCEEDED", "FAILED"):
            if job_status == "FAILED":
                reason = job.get("statusReason", "unknown")
                logger.error(f"Job {job_id} failed. Reason: {reason}")
            return
        sleep(30)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--env_var_prefix",
    type=str,
    required=True,
)
@click.option(
    "--batch_job_name_prefix",
    type=str,
    required=True,
)
@click.option(
    "--observe",
    is_flag=True,
    default=False,
    show_default=True,
)
def main(config: str, env_var_prefix: str, batch_job_name_prefix: str, observe: bool) -> None:
    cfg = load_yaml_config(config)
    params = extract_batch_params(cfg)

    environment = build_environment(cfg, env_var_prefix)
    log_environment_vars(environment)

    job_name = f"{batch_job_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_id = submit_batch_job(boto3.client("batch"), params, job_name, environment)
    logger.info(f"Submitted batch job '{job_name}' with ID: {job_id}")

    if observe:
        observe_batch_job(job_id)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
