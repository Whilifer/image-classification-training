from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "CIFAR10-classification"
METRIC_NAME = "best_validation_accuracy"


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

    print("Best run:")
    print("Run ID:", best_run.info.run_id)
    print(
        "Validation accuracy:",
        best_run.data.metrics[METRIC_NAME],
    )


if __name__ == "__main__":
    main()
