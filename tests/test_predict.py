from pathlib import Path


def test_predict(client):
    image_path = Path("samples/airplane_003.png")

    with image_path.open("rb") as file:
        response = client.post(
            "/predict/",
            files={
                "file": (
                    image_path.name,
                    file,
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["class_id"] == 0
    assert body["class_name"] == "airplane"

    assert isinstance(body["confidence"], float)
    assert 0 <= body["confidence"] <= 1


def test_predict_without_file(client):
    response = client.post("/predict/")

    assert response.status_code == 422


def test_predict_text_file(client):
    response = client.post(
        "/predict/",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_invalid_image(client):
    response = client.post(
        "/predict/",
        files={
            "file": (
                "broken.png",
                b"not an image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_predict_empty_file(client):
    response = client.post(
        "/predict/",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
