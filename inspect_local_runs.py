import mlflow
from mlflow.tracking import MlflowClient

# 1. Read from Local DB
local_uri = "sqlite:///experiments/mlflow.db"
print(f"Reading from: {local_uri}")

mlflow.set_tracking_uri(local_uri)
client = MlflowClient(tracking_uri=local_uri)

# Get all experiments
experiments = client.search_experiments()
print(f"Found {len(experiments)} experiments")

for exp in experiments:
    print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
    runs = client.search_runs(exp.experiment_id)
    print(f"Found {len(runs)} runs")
    
    for run in runs:
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Status: {run.info.status}")
        print(f"  Metrics: {len(run.data.metrics)}")
        print(f"  Params: {len(run.data.params)}")
        if len(run.data.metrics) > 0:
            print(f"  Example metric: {list(run.data.metrics.items())[0]}")
