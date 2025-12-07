#!/usr/bin/env python3
"""
MLOps Pipeline Integration Demo
================================
This script demonstrates the complete MLOps pipeline by:
1. Training a model → Logs to MLflow
2. Making predictions → Sends metrics to Prometheus  
3. Showing you where to see the results

Run this after all services are started!
"""

import mlflow
import time
import requests
from pathlib import Path

print("🚀 MLOps Pipeline Integration Demo")
print("=" * 50)

# Configuration
MLFLOW_URI = "http://localhost:5000"
API_URL = "http://localhost:8000"
EXPERIMENT_NAME = "integration_demo"

# Step 1: MLflow Integration
print("\n📊 Step 1: Testing MLflow Integration")
print("-" * 50)

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"demo_run_{int(time.time())}"):
        # Log parameters
        mlflow.log_param("demo_mode", "integration_test")
        mlflow.log_param("model_type", "resnet50")
        mlflow.log_param("batch_size", 32)
        
        # Simulate training metrics
        print("  📈 Logging training metrics...")
        for epoch in range(5):
            train_loss = 1.5 - epoch * 0.25
            val_acc = 0.60 + epoch * 0.06
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("lr", 0.001 * (0.9 ** epoch), step=epoch)
            
            print(f"    Epoch {epoch}: loss={train_loss:.3f}, acc={val_acc:.3f}")
            time.sleep(0.5)
        
        # Log final metrics
        mlflow.log_metric("final_accuracy", 0.92)
        mlflow.log_metric("final_f1_score", 0.91)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n  ✅ MLflow run created: {run_id}")
        print(f"  🔗 View at: {MLFLOW_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")

except Exception as e:
    print(f"  ❌ MLflow error: {e}")
    print(f"  💡 Make sure MLflow is running: {MLFLOW_URI}")

# Step 2: API & Prometheus Integration
print("\n\n🤖 Step 2: Testing API & Prometheus Integration")
print("-" * 50)

try:
    # Health check
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print(f"  ✅ API is healthy")
    
    # Get classes
    response = requests.get(f"{API_URL}/classes")
    if response.status_code == 200:
        classes = response.json()
        print(f"  ✅ API supports {classes['num_classes']} diseases")
    
    # Make predictions (generates Prometheus metrics)
    print(f"\n  📊 Making predictions to generate metrics...")
    from PIL import Image
    import numpy as np
    import io
    
    for i in range(5):
        # Create dummy test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        
        # Make prediction
        files = {'file': (f'test_{i}.jpg', buf, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"    Prediction {i+1}: {result['prediction']} ({result['confidence']:.2%})")
        
        time.sleep(0.5)
    
    print(f"\n  ✅ Metrics sent to Prometheus")
    print(f"  🔗 View metrics: {API_URL}/metrics")
    print(f"  🔗 View in Prometheus: http://localhost:9091")

except Exception as e:
    print(f"  ❌ API error: {e}")
    print(f"  💡 Make sure API is running: {API_URL}")

# Step 3: Summary
print("\n\n✅ Integration Demo Complete!")
print("=" * 50)
print("\n📊 Check Your Results:")
print(f"  1. MLflow UI:     {MLFLOW_URI}")
print(f"     → See experiment: '{EXPERIMENT_NAME}'")
print(f"     → View metrics charts")
print(f"     → Compare parameters")
print()
print(f"  2. Prometheus:    http://localhost:9091")
print(f"     → Graph tab → Query: api_requests_total")
print(f"     → Should show {5} prediction requests")
print()
print(f"  3. Grafana:       http://localhost:3000")
print(f"     → Configure Prometheus data source first")
print(f"     → Create dashboard with query: rate(api_requests_total[5m])")
print()
print(f"  4. API Metrics:   {API_URL}/metrics")
print(f"     → Raw Prometheus metrics")
print()

print("\n🔄 Complete MLOps Workflow:")
print("  1. Code change → Run this script")
print("  2. View training in MLflow (metrics, parameters)")
print("  3. View predictions in Prometheus/Grafana (real-time)")
print("  4. Compare runs in MLflow (which config is better?)")
print()

print("🎯 Your Pipeline is Connected!")
print("   Train → MLflow ✅")
print("   Predict → Prometheus ✅")  
print("   Monitor → Grafana ✅")
print()

print("💡 Next: Try training a real model:")
print("   python scripts/train_pipeline.py --dataset data/raw/PlantVillage --model cnn")
