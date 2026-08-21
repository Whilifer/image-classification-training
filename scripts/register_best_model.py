import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "CIFAR10-classification"
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

    print("Best run:")
    print("Run ID:", run_id)
    print("Validation accuracy:", accuracy)

    model_uri = f"runs:/{run_id}/model"

    print("Model URI:", model_uri)

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print(f"Registered model: {result.name} version {result.version}")


if __name__ == "__main__":
    main()
