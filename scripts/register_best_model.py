import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "CIFAR10-classification-docker-v2"
METRIC_NAME = "best_validation_accuracy"
MODEL_NAME = "CIFARClassifier"


def main():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            f"metrics.{METRIC_NAME} DESC",
        ],
    )

    if not runs:
        raise RuntimeError("No runs found")

    best_run = runs[0]

    run_id = best_run.info.run_id
    accuracy = best_run.data.metrics[METRIC_NAME]

    logger.info("Best run:")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Validation accuracy: {accuracy}")

    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Model URI: {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        result.version,
    )

    logger.info(
        "Registered model: %s version %s as @champion",
        result.name,
        result.version,
    )


if __name__ == "__main__":
    main()
