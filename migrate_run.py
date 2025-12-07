import mlflow
from mlflow.tracking import MlflowClient
import time

# Configuration
LOCAL_URI = "sqlite:///experiments/mlflow.db"
REMOTE_URI = "http://localhost:5000"
EXPERIMENT_NAME = "plant_disease_detection"

print(f"🚀 Migrating run from {LOCAL_URI} to {REMOTE_URI}")

# 1. Connect to Local DB
local_client = MlflowClient(tracking_uri=LOCAL_URI)
local_exp = local_client.get_experiment_by_name(EXPERIMENT_NAME)
if not local_exp:
    print("❌ Local experiment not found")
    exit(1)

# Find the successful run
runs = local_client.search_runs(local_exp.experiment_id, filter_string="status = 'FINISHED'")
# Sort by start time descending to get the latest
runs.sort(key=lambda x: x.info.start_time, reverse=True)

if not runs:
    print("❌ No FINISHED runs found locally")
    exit(1)

target_run = runs[0]
print(f"✅ Found local run: {target_run.info.run_id}")
print(f"   Start time: {time.ctime(target_run.info.start_time/1000)}")
print(f"   Metrics: {len(target_run.data.metrics)}")
print(f"   Params: {len(target_run.data.params)}")

# 2. Connect to Remote Server
mlflow.set_tracking_uri(REMOTE_URI)
remote_client = MlflowClient(tracking_uri=REMOTE_URI)

# Ensure experiment exists remotely
try:
    remote_exp_id = remote_client.create_experiment(EXPERIMENT_NAME)
except:
    remote_exp = remote_client.get_experiment_by_name(EXPERIMENT_NAME)
    remote_exp_id = remote_exp.experiment_id

print(f"✅ Connected to remote experiment ID: {remote_exp_id}")

# 3. Re-create the run remotely
with mlflow.start_run(experiment_id=remote_exp_id, run_name="migrated_training_run") as run:
    print(f"🚀 Created new remote run: {run.info.run_id}")
    
    # Log params
    print("   Logging params...")
    for key, value in target_run.data.params.items():
        mlflow.log_param(key, value)
        
    # Log metrics
    print("   Logging metrics...")
    for key, value in target_run.data.metrics.items():
        mlflow.log_metric(key, value)
        
    # Log tags
    print("   Logging tags...")
    for key, value in target_run.data.tags.items():
        mlflow.set_tag(key, value)
        
    mlflow.set_tag("migration_source", "local_db")
    mlflow.set_tag("original_run_id", target_run.info.run_id)

    # Upload Artifacts
    print("   Uploading model artifact...")
    try:
        mlflow.log_artifact("models/production/model.ckpt", artifact_path="model")
        print("   ✅ Model uploaded successfully")
    except Exception as e:
        print(f"   ⚠️ Could not upload model: {e}")

    # Upload Class Mapping
    print("   Uploading class mapping...")
    try:
        mlflow.log_artifact("data/class_mapping.json", artifact_path="data")
        print("   ✅ Class mapping uploaded successfully")
    except Exception as e:
        print(f"   ⚠️ Could not upload class mapping: {e}")

print("\n✅ Migration Complete!")
print(f"View it at: {REMOTE_URI}/#/experiments/{remote_exp_id}/runs/{run.info.run_id}")
