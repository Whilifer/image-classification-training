import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")
c = mlflow.MlflowClient()
versions = c.search_model_versions("name='CIFARClassifier'")
print([(m.version, m.current_stage, m.aliases) for m in versions])
