import logging

import mlflow

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5001")

mlflow.set_experiment("artifact-test")

with mlflow.start_run() as run:
    logger.info(f"RUN: {run.info.run_id}")
    logger.info(f"ARTIFACT: {run.info.artifact_uri}")

    mlflow.log_text(
        "test",
        "test.txt",
    )
