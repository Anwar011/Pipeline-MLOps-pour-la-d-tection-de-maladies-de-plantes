# 🎯 Repository Walkthrough - Plant Disease Detection MLOps

## What This Repository Does

This is a **complete MLOps pipeline** that enables automatic detection of plant diseases from images. It's built for production use with all the infrastructure needed to:

- Train deep learning models
- Track experiments
- Serve predictions via API
- Deploy to Kubernetes
- Monitor performance

---

## Repository Structure Explained

```
Pipeline-MLOps-pour-la-d-tection-de-maladies-de-plantes/
│
├── 📄 config.yaml                 # Central configuration (ALL hyperparameters)
├── 📄 requirements.txt            # Python dependencies
│
├── 📂 src/                        # Core source code
│   ├── data_preprocessing.py     # Load & augment images
│   ├── models.py                 # CNN & ViT architectures
│   ├── train.py                  # Training with MLflow
│   └── api.py                    # FastAPI inference server
│
├── 📂 scripts/                    # Entry points
│   ├── train_pipeline.py         # Run full training
│   └── run_api.py                # Start API server
│
├── 📂 data/                       # Data storage
│   ├── raw/                      # Original images
│   ├── processed/                # Preprocessed data
│   └── class_mapping.json        # Class ID → Name mapping
│
├── 📂 models/                     # Trained models
│   ├── checkpoints/              # Training checkpoints
│   └── production/               # Production-ready models
│
├── 📂 docker/                     # Containerization
│   ├── Dockerfile                # API container definition
│   └── docker-compose.yml        # Local development stack
│
├── 📂 k8s/                        # Kubernetes manifests
│   ├── namespace.yaml            # Isolate resources
│   ├── deployment.yaml           # API deployment (3 replicas)
│   ├── service.yaml              # LoadBalancer
│   ├── storage.yaml              # Persistent volumes
│   └── hpa.yaml                  # Autoscaling rules
│
├── 📂 monitoring/                 # Observability
│   ├── prometheus.yml            # Metrics collection config
│   └── grafana/                  # Dashboard definitions
│
├── 📂 .github/workflows/          # CI/CD automation
│   └── ci.yml                    # Test, build, deploy pipeline
│
└── 📂 notebooks/                  # Jupyter demos
    └── demo_pipeline.ipynb       # Interactive walkthrough
```

---

## The Journey of Data Through the System

### Phase 1: Data Preparation

**Input:** Raw images in `data/raw/PlantVillage/`

```
PlantVillage/
├── Tomato_Early_blight/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Tomato_healthy/
│   └── ...
└── Potato_Late_blight/
    └── ...
```

**What happens:**

1. **`DataPreprocessor`** scans directories
2. Creates class mapping: `{0: 'Tomato_Early_blight', 1: 'Tomato_healthy', ...}`
3. Splits data: 70% train, 20% val, 10% test (stratified)
4. Applies augmentation (rotation, flip, brightness, etc.)
5. Creates PyTorch DataLoaders

**Output:** Ready-to-train batches of tensors

---

### Phase 2: Model Training

**Entry point:** `python scripts/train_pipeline.py --dataset data/raw/PlantVillage --model cnn`

**What happens:**

```mermaid
graph LR
    A[Load Config] --> B[Create DataLoaders]
    B --> C[Initialize Model]
    C --> D[Setup MLflow]
    D --> E[Training Loop]
    E --> F[Validation]
    F --> G[Save Best Model]
    G --> H[Test Evaluation]
    H --> I[Log to MLflow]
```

**Step-by-step:**

1. **Load `config.yaml`**: Get all hyperparameters
2. **Preprocess data**: Use `DataPreprocessor`
3. **Create model**: Choose CNN (ResNet50/EfficientNet/VGG16) or ViT
4. **Setup MLflow**: Create experiment, start run
5. **Train**: PyTorch Lightning handles the loop
   - Forward pass
   - Compute loss (CrossEntropy)
   - Backward pass
   - Update weights (Adam optimizer)
   - Validate every epoch
   - Save best checkpoint
6. **Test**: Evaluate on held-out test set
7. **Log everything**: Metrics, parameters, model artifacts

**Output:** Trained model in `models/checkpoints/` and MLflow registry

---

### Phase 3: Model Serving

**Entry point:** `python scripts/run_api.py` or `docker-compose up`

**What happens:**

```python
# API startup
1. Load config.yaml
2. Load class_mapping.json
3. Load trained model checkpoint
4. Initialize FastAPI app
5. Start Uvicorn server on port 8000
```

**API endpoints available:**

```bash
GET  /               # API info
GET  /health         # Status check
POST /predict        # Single image prediction
POST /predict_batch  # Batch predictions
GET  /classes        # List supported diseases
GET  /metrics        # Prometheus metrics
```

**Prediction flow:**

```
Client sends JPEG → API receives → Preprocess (resize, normalize) 
  → Model inference → Softmax → Top prediction + confidence → JSON response
```

---

### Phase 4: Deployment

#### Docker Deployment

```bash
docker-compose up -d
```

**Services launched:**

1. **plant-disease-api** (port 8000) - Main API
2. **mlflow-server** (port 5000) - Experiment tracking UI
3. **prometheus** (port 9090) - Metrics scraping
4. **grafana** (port 3000) - Dashboards
5. **redis** (port 6379) - Optional caching

#### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

**Resources created:**

1. **Namespace** (`mlops`) - Isolation
2. **PersistentVolumeClaims** - Model & data storage
3. **Deployment** (3 replicas) - API pods with:
   - CPU: 1-2 cores
   - Memory: 2-4 GB
   - GPU: 1 (if available)
   - Health checks (liveness + readiness)
4. **Service** (LoadBalancer) - External access
5. **HorizontalPodAutoscaler** - Auto-scale 3-10 pods based on CPU

---

### Phase 5: Monitoring

**Metrics collected:**

```python
# From API
predictions_total                    # Counter
prediction_errors_total              # Counter
prediction_duration_seconds          # Histogram
prediction_confidence                # Gauge

# From Kubernetes
container_cpu_usage_seconds_total
container_memory_usage_bytes
kube_deployment_status_replicas
```

**Prometheus scrapes** these metrics every 15 seconds.

**Grafana visualizes** with dashboards showing:
- Request rate & latency
- Error rate & types
- Prediction confidence distribution
- Resource utilization
- Pod health status

---

## Key Files Deep Dive

### `config.yaml` - The Control Center

```yaml
# Data configuration
data:
  batch_size: 32           # How many images per training batch
  image_size: [224, 224]   # All images resized to this
  train_split: 0.7         # 70% for training
  
# Model configuration  
model:
  architecture: "resnet50"  # Which CNN backbone
  num_classes: 15          # Number of disease types
  pretrained: true         # Use ImageNet weights
  
# Training configuration
training:
  epochs: 5                # Training iterations (increase for production)
  learning_rate: 0.001     # How fast model learns
  optimizer: "adam"        # Optimization algorithm
```

**Everything is configurable here** - no hardcoded values!

---

### `src/data_preprocessing.py` - Data Pipeline

**Two main classes:**

#### `PlantDiseaseDataset` (PyTorch Dataset)

```python
dataset = PlantDiseaseDataset(
    image_paths=['path1.jpg', 'path2.jpg'],
    labels=[0, 1],
    transform=augmentation_pipeline
)

# When you iterate:
for image_tensor, label in dataset:
    # image_tensor: (3, 224, 224) torch.Tensor
    # label: integer class ID
```

#### `DataPreprocessor` (Orchestrator)

```python
preprocessor = DataPreprocessor('config.yaml')
result = preprocessor.create_data_loaders('data/raw/PlantVillage')

# Returns:
{
    'data_loaders': {
        'train': DataLoader,  # Shuffled, augmented
        'val': DataLoader,     # Not shuffled
        'test': DataLoader     # Not shuffled
    },
    'class_names': ['Disease1', 'Disease2', ...],
    'num_classes': 15
}
```

---

### `src/models.py` - Neural Networks

**Two architectures implemented:**

#### CNN (Convolutional Neural Network)

```python
model = PlantDiseaseCNN(config_path='config.yaml')

# Architecture:
# Input (3x224x224) → Backbone (ResNet50/EfficientNet/VGG) 
#   → Global Pooling → Classifier → Output (num_classes)

# Features:
- Transfer learning from ImageNet
- Freezable backbone
- Custom classifier head
```

#### ViT (Vision Transformer)

```python
model = PlantDiseaseViT(config_path='config.yaml')

# Architecture:
# Input → Patch Embedding → Transformer Encoder → Classification Head

# Features:
- Self-attention mechanism
- Position embeddings
- 12-layer transformer
```

**Both inherit from `pytorch_lightning.LightningModule`** for automatic training.

---

### `src/train.py` - Training Engine

**Key functions:**

#### `setup_mlflow(config)` - Experiment Tracking

```python
mlflow.set_tracking_uri('sqlite:///experiments/mlflow.db')
mlflow.set_experiment('plant_disease_detection')
```

#### `train_model(model_type, dataset_path, config_path)` - Main Training

```python
# 1. Prepare data
data = DataPreprocessor().create_data_loaders(dataset_path)

# 2. Create model
model = create_model(model_type, config_path)

# 3. Setup trainer
trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[checkpoint, early_stopping],
    logger=mlflow_logger
)

# 4. Train
trainer.fit(model, data['train'], data['val'])

# 5. Test
trainer.test(model, data['test'])
```

---

### `src/api.py` - Inference Server

**Main class:**

```python
class PlantDiseaseInferenceAPI:
    def __init__(self):
        # Load config, model, class mapping
        
    def preprocess_image(self, image_bytes):
        # PIL load → Resize → Normalize → Tensor
        
    def predict(self, image_tensor):
        # Model forward pass → Softmax → Top prediction
```

**FastAPI endpoints:**

```python
@app.post("/predict")
async def predict_disease(file: UploadFile):
    # 1. Read file bytes
    contents = await file.read()
    
    # 2. Preprocess
    tensor = inference_api.preprocess_image(contents)
    
    # 3. Predict
    result = inference_api.predict(tensor)
    
    # 4. Return JSON
    return {
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "probabilities": result['probabilities']
    }
```

---

## Configuration Deep Dive

### Adjusting for Your Needs

**Want faster training?**
```yaml
data:
  batch_size: 64  # Increase batch size (needs more RAM)
  num_workers: 8  # More parallel data loading
```

**Need better accuracy?**
```yaml
model:
  architecture: "vit"  # Use Vision Transformer
training:
  epochs: 100         # Train longer
  learning_rate: 0.0001  # Slower, more careful learning
```

**Less memory available?**
```yaml
data:
  batch_size: 8   # Smaller batches
model:
  architecture: "efficientnet"  # Lighter model
```

**Production deployment?**
```yaml
training:
  epochs: 50
  save_best_only: true
  early_stopping_patience: 10  # Stop if not improving
  
kubernetes:
  replicas: 5        # More pods
  max_replicas: 20   # Higher autoscaling limit
```

---

## Practical Examples

### Example 1: Complete Training Run

```bash
# 1. Ensure dataset is ready
ls data/raw/PlantVillage/  # Should see class folders

# 2. Verify config
cat config.yaml  # Check settings

# 3. Train model
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model cnn

# Output:
# 🚀 Démarrage du pipeline MLOps...
# 📊 Étape 1: Préparation des données
# ✅ Données préparées: 15 classes trouvées
# 🤖 Étape 2: Entraînement du modèle
# Epoch 1/5: train_loss=1.234, val_acc=0.65
# Epoch 2/5: train_loss=0.856, val_acc=0.78
# ...
# ✅ Entraînement terminé
# 📈 Étape 3: Évaluation
# Test Accuracy: 0.92
# 🎉 Pipeline terminé!

# 4. View in MLflow
firefox http://localhost:5000  # See all experiments
```

---

### Example 2: Making Predictions

```python
import requests
from PIL import Image

# Take a photo of a plant leaf
image_path = "my_tomato_leaf.jpg"

# Send to API
with open(image_path, 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()

# Output:
# {
#     "prediction": "Tomato_Early_blight",
#     "confidence": 0.94,
#     "probabilities": {
#         "Tomato_Early_blight": 0.94,
#         "Tomato_Late_blight": 0.03,
#         "Tomato_healthy": 0.02,
#         ...
#     },
#     "inference_time_ms": 45.2
# }

print(f"Your plant has: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

---

### Example 3: Monitoring in Production

```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. View Prometheus metrics
curl http://localhost:8000/metrics
# Output:
# predictions_total 1523
# prediction_duration_seconds_bucket{le="0.05"} 1402
# prediction_errors_total 3

# 3. Open Grafana
firefox http://localhost:3000
# Login: admin/admin
# View dashboards:
# - API Performance
# - ML Model Metrics
# - Infrastructure Health

# 4. Check Kubernetes pods
kubectl get pods -n mlops
# NAME                                  READY   STATUS
# plant-disease-api-7d9f8b6c4-abcd1     1/1     Running
# plant-disease-api-7d9f8b6c4-efgh2     1/1     Running
# plant-disease-api-7d9f8b6c4-ijkl3     1/1     Running

# 5. View logs
kubectl logs -f deployment/plant-disease-api -n mlops
```

---

## Common Workflows

### Workflow: Update Model in Production

```bash
# 1. Train new model
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model cnn

# 2. Evaluate performance
# Check MLflow UI for metrics

# 3. Copy best model to production
cp models/checkpoints/epoch=49-val_acc=0.95.ckpt \
   models/production/model.ckpt

# 4. Rebuild Docker image
docker build -f docker/Dockerfile -t plant-disease-api:v2 .

# 5. Update Kubernetes deployment
kubectl set image deployment/plant-disease-api \
  api=plant-disease-api:v2 \
  -n mlops

# 6. Monitor rollout
kubectl rollout status deployment/plant-disease-api -n mlops

# 7. Verify
curl http://<service-ip>/health
```

---

### Workflow: Debug Low Accuracy

```bash
# 1. Check data quality
python -c "
from src.data_preprocessing import DataPreprocessor
prep = DataPreprocessor()
result = prep.create_data_loaders('data/raw/PlantVillage')
print(f'Classes: {result[\"class_names\"]}')
print(f'Train size: {len(result[\"data_loaders\"][\"train\"].dataset)}')
print(f'Val size: {len(result[\"data_loaders\"][\"val\"].dataset)}')
"

# 2. Check for class imbalance
python -c "
import os
for cls in os.listdir('data/raw/PlantVillage'):
    count = len(os.listdir(f'data/raw/PlantVillage/{cls}'))
    print(f'{cls}: {count} images')
"

# 3. Try different architecture
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model vit  # Instead of cnn

# 4. Adjust hyperparameters in config.yaml
# - Increase epochs
# - Lower learning rate
# - Add more augmentation

# 5. Retrain and compare in MLflow
```

---

## Testing the Components

```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test model creation
python -c "
from src.models import create_model
model = create_model('cnn')
print(f'Model created: {model.__class__.__name__}')
"

# Test API locally
python src/api.py &
sleep 5
curl http://localhost:8000/health
kill %1

# Test with Docker
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml logs -f
```

---

## Summary

This pipeline provides **everything you need** for production ML:

✅ **Data handling** - Preprocessing, augmentation, splitting  
✅ **Model training** - Multiple architectures, transfer learning  
✅ **Experiment tracking** - MLflow for reproducibility  
✅ **API serving** - FastAPI for real-time inference  
✅ **Containerization** - Docker for portability  
✅ **Orchestration** - Kubernetes for scaling  
✅ **CI/CD** - GitHub Actions for automation  
✅ **Monitoring** - Prometheus + Grafana for observability  

**Start here**: `quick_start_guide.md`  
**Go deeper**: `comprehensive_analysis.md`  
**Expert level**: `technical_deep_dive.md`

🎉 **You now have a production-ready MLOps system!**
