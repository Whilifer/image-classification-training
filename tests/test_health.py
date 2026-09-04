def test_health(client):
    response = client.get("/health/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == "CIFARClassifier"
    assert body["model_alias"] == "champion"
    assert body["device"] == "cpu"
