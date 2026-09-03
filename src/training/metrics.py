import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision_macro": precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
        ).tolist(),
    }
