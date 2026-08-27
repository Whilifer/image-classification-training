import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")

mlflow.set_experiment("artifact-test")

with mlflow.start_run() as run:
    print("RUN:", run.info.run_id)
    print("ARTIFACT:", run.info.artifact_uri)

    mlflow.log_text(
        "test",
        "test.txt",
    )
