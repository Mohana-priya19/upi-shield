import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

run_id = "e5acc91ded824c789e07446a58214c4e"

result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="upi-shield-xgboost"
)

print(f"Model registered: {result.name}")
print(f"Version: {result.version}")

client.transition_model_version_stage(
    name="upi-shield-xgboost",
    version=result.version,
    stage="Staging"
)

print(f"Model transitioned to: Staging")