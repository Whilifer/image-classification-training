import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from logging_config import setup_logging
from src.config import TrainConfig

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "CIFAR10-classification-docker-v2"
METRIC_NAME = "best_validation_accuracy"
MODEL_NAME = "CIFARClassifier"
MODEL_ALIAS = "champion"


def main():
    setup_logging()

    config = TrainConfig.from_yaml("configs/train.yaml")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(f"No runs with metric '{METRIC_NAME}' found")

    best_run = runs[0]
    run_id = best_run.info.run_id
    candidate_accuracy = best_run.data.metrics[METRIC_NAME]

    logger.info("Best run:")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Validation accuracy: {candidate_accuracy:.4f}")

    model_versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}' AND run_id='{run_id}'"
    )

    if not model_versions:
        raise RuntimeError(f"No registered model version found for run '{run_id}'")

    candidate_version = model_versions[0].version
    logger.info(f"Candidate model version: {candidate_version}")

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        champion_version = champion.version
        champion_run_id = champion.run_id

        champion_run = client.get_run(champion_run_id)
        champion_accuracy = champion_run.data.metrics.get(METRIC_NAME, 0.0)

        logger.info(f"Current champion version: {champion_version}")
        logger.info(f"Current champion accuracy: {champion_accuracy:.4f}")
    except MlflowException:
        champion_version = None
        champion_accuracy = None
        logger.info("No current champion found")

    if champion_accuracy is None or candidate_accuracy > champion_accuracy:
        client.set_registered_model_alias(
            MODEL_NAME,
            MODEL_ALIAS,
            candidate_version,
        )
        logger.info(f"Model version {candidate_version} is now '{MODEL_ALIAS}'")
    else:
        logger.info(f"Champion remains version {champion_version}")


if __name__ == "__main__":
    main()
