# 🔬 Technical Deep Dive - Plant Disease Detection MLOps

## Advanced Topics & Implementation Details

---

## 1. Model Architecture Internals

### CNN Architecture Deep Dive

#### ResNet50 Adaptation
```python
# Original ResNet50 structure
Input (3, 224, 224)
  ↓
Conv1 (7x7, stride 2) + BN + ReLU
  ↓
MaxPool (3x3, stride 2)
  ↓
ResBlock_1 (3 bottleneck blocks) - 64 channels
  ↓
ResBlock_2 (4 bottleneck blocks) - 128 channels
  ↓
ResBlock_3 (6 bottleneck blocks) - 256 channels
  ↓
ResBlock_4 (3 bottleneck blocks) - 512 channels
  ↓
AvgPool (7x7)
  ↓
FC (2048 → num_classes)  # <-- Modified for our task
```

**Bottleneck Block Structure:**
```python
x → Conv1x1(reduce) → BN → ReLU
   → Conv3x3(process) → BN → ReLU
   → Conv1x1(expand) → BN
   → (+ skip connection) → ReLU
```

**Code Implementation:**
```python
def _create_backbone(self):
    if self.architecture == "resnet50":
        backbone = models.resnet50(pretrained=True)
        # Remove final FC layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone_output_size = 2048
    
    if self.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    return backbone, backbone_output_size
```

---

### Vision Transformer Deep Dive

#### Patch Embedding Process
```python
Input Image: (3, 224, 224)
  ↓
Split into patches: 16x16 = 196 patches of size (3, 16, 16)
  ↓
Flatten each patch: 196 × 768 dimensions
  ↓
Add [CLS] token: (1 + 196) × 768
  ↓
Add positional embeddings: (197) × 768
```

**Self-Attention Mechanism:**
```python
Q = x @ W_q  # Query projection
K = x @ W_k  # Key projection
V = x @ W_v  # Value projection

Attention = softmax((Q @ K.T) / sqrt(d_k)) @ V

# Multi-head: Split into 12 heads
# head_dim = 768 / 12 = 64
```

**Why ViT Works for Plants:**
- Global receptive field (sees entire image)
- Learns spatial relationships between diseased regions
- Better at subtle texture patterns than CNNs
- Captures long-range dependencies

---

## 2. PyTorch Lightning Patterns

### Lightning Module Lifecycle

```python
# 1. Initialization
model = PlantDiseaseCNN(config_path)

# 2. Training loop (automated by Trainer)
for epoch in range(num_epochs):
    # Training
    for batch in train_loader:
        loss = model.training_step(batch, batch_idx)  # You define
        optimizer.zero_grad()                          # Lightning handles
        loss.backward()                                # Lightning handles
        optimizer.step()                               # Lightning handles
    
    # Validation
    for batch in val_loader:
        metrics = model.validation_step(batch, batch_idx)
    
    # Callbacks
    checkpoint_callback.on_epoch_end()
    early_stopping.on_epoch_end()

# 3. Testing
test_results = trainer.test(model, test_loader)
```

### Custom Metrics Implementation

```python
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

class PlantDiseaseCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Define metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        # Log
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        return loss
```

---

## 3. Advanced Data Preprocessing

### Custom Augmentation Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Production-grade augmentation
def get_advanced_augmentation():
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        
        # Color transforms (careful - diseases have color signatures)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        
        # Quality degradation (simulate real-world conditions)
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ISONoise(p=0.2),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        
        # Weather/lighting simulation
        A.RandomShadow(p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
        
        # Normalization (ALWAYS last)
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
```

### Mixed Precision Training

```python
# Enable automatic mixed precision (AMP)
trainer = pl.Trainer(
    precision="16-mixed",  # FP16 for speed, FP32 for stability
    accelerator="gpu"
)

# Benefits:
# - 2-3x faster training
# - 50% less memory (allows larger batches)
# - Minimal accuracy loss (<0.5%)
```

---

## 4. MLflow Advanced Usage

### Custom Metrics Logging

```python
import mlflow

# Log confusion matrix as image
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix.png')
mlflow.log_artifact('confusion_matrix.png')

# Log class-wise metrics
for i, class_name in enumerate(class_names):
    precision = precision_per_class[i]
    recall = recall_per_class[i]
    mlflow.log_metric(f'precision_{class_name}', precision)
    mlflow.log_metric(f'recall_{class_name}', recall)

# Log learning curves
for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    mlflow.log_metric('train_loss', train_loss, step=epoch)
    mlflow.log_metric('val_loss', val_loss, step=epoch)
```

### Model Registry Usage

```python
# Register model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="PlantDiseaseClassifier"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="PlantDiseaseClassifier",
    version=3,
    stage="Production"
)

# Load production model
model_uri = "models:/PlantDiseaseClassifier/Production"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

---

## 5. FastAPI Advanced Patterns

### Request Validation

```python
from pydantic import BaseModel, Field, validator

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted disease class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    inference_time_ms: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    images: list[str]  # Base64 encoded
    
    @validator('images')
    def validate_images(cls, v):
        if len(v) > 16:
            raise ValueError('Maximum 16 images per batch')
        return v
```

### Async Processing

```python
from fastapi import BackgroundTasks

@app.post("/predict_async")
async def predict_async(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Return immediately
    task_id = str(uuid.uuid4())
    
    # Process in background
    background_tasks.add_task(
        process_prediction,
        file,
        task_id
    )
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # Retrieve from cache/database
    result = redis_client.get(task_id)
    return json.loads(result)
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict_disease(request: Request, file: UploadFile):
    # ... prediction logic
```

---

## 6. Kubernetes Advanced Configuration

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: mlops-quota
  namespace: mlops
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    requests.nvidia.com/gpu: "2"
    persistentvolumeclaims: "5"
    pods: "20"
```

### Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: plant-disease-pdb
  namespace: mlops
spec:
  minAvailable: 2  # Always keep 2 pods running
  selector:
    matchLabels:
      app: plant-disease-api
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: plant-disease-vpa
  namespace: mlops
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: plant-disease-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
```

---

## 7. Performance Optimization

### Model Optimization Techniques

**1. Quantization (INT8)**
```python
import torch.quantization as quant

# Post-training quantization
model_fp32 = trained_model
model_fp32.eval()

# Fuse layers
model_fused = torch.quantization.fuse_modules(
    model_fp32,
    [['conv', 'bn', 'relu']]
)

# Quantize
model_int8 = quant.quantize_dynamic(
    model_fused,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Result: 4x smaller, 2-4x faster, <1% accuracy loss
```

**2. ONNX Export**
```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "plant_disease_model.onnx",
    export_params=True,
    opset_version=13,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

# Use with ONNX Runtime for faster inference
```

**3. TorchScript**
```python
# JIT compilation
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load and use
loaded = torch.jit.load("model_scripted.pt")
output = loaded(input_tensor)
```

---

### API Optimization

**1. Model Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model():
    model = load_model()
    model.eval()
    return model

# Model loaded once, cached for subsequent requests
```

**2. Batch Inference**
```python
# Instead of processing images one-by-one
# Accumulate small batches

from collections import deque
import threading

batch_queue = deque(maxlen=16)
batch_lock = threading.Lock()

def add_to_batch(image_tensor, callback):
    with batch_lock:
        batch_queue.append((image_tensor, callback))
        
        if len(batch_queue) >= 8:  # Batch size
            process_batch()

def process_batch():
    images = torch.stack([item[0] for item in batch_queue])
    results = model(images)
    
    for (_, callback), result in zip(batch_queue, results):
        callback(result)
    
    batch_queue.clear()
```

---

## 8. Monitoring Deep Dive

### Custom Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Counter: Cumulative metric
class_predictions = Counter(
    'predictions_by_class',
    'Predictions per class',
    ['class_name']
)
class_predictions.labels(class_name='Tomato_Early_blight').inc()

# Histogram: Distribution of values
inference_duration = Histogram(
    'model_inference_seconds',
    'Model inference time',
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
)
with inference_duration.time():
    prediction = model(image)

# Gauge: Current value
active_requests = Gauge('active_prediction_requests', 'Active requests')
active_requests.inc()  # Increment
# ... do work ...
active_requests.dec()  # Decrement

# Info: Static metadata
model_info = Info('model_metadata', 'Model information')
model_info.info({
    'version': '1.0.0',
    'architecture': 'resnet50',
    'num_classes': '15'
})
```

### Data Drift Detection

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

# Reference data (training set)
reference_df = pd.DataFrame({
    'feature_1': train_features[:, 0],
    'feature_2': train_features[:, 1],
    # ...
})

# Production data (last 1000 predictions)
production_df = pd.DataFrame({
    'feature_1': prod_features[:, 0],
    'feature_2': prod_features[:, 1],
    # ...
})

# Detect drift
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference_df, production_df)
dashboard.save('drift_report.html')

# Alert if drift detected
if dashboard.drift_detected:
    send_alert("Data drift detected! Model retraining recommended.")
```

---

## 9. Production Best Practices

### Graceful Shutdown

```python
import signal
import sys

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)
    
    def request_shutdown(self, *args):
        print("Shutdown requested, finishing current requests...")
        self.shutdown_requested = True

shutdown_handler = GracefulShutdown()

@app.on_event("shutdown")
async def shutdown_event():
    # Save state
    save_metrics()
    close_connections()
    logger.info("Graceful shutdown complete")
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "disk_space": check_disk_space(),
        "memory_usage": get_memory_usage()
    }
    
    healthy = all(checks.values())
    
    return {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Logging Strategy

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)

# Structured logging
logger.info("Prediction completed", extra={
    "prediction": "Tomato_Early_blight",
    "confidence": 0.94,
    "inference_time_ms": 45.2
})
```

---

## 10. Custom Model Development

### Creating a New Architecture

```python
class CustomPlantCNN(pl.LightningModule):
    def __init__(self, num_classes=15):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        features = features * attention_weights
        
        return self.classifier(features)
```

---

## 11. CI/CD Advanced Patterns

### Multi-Environment Deployment

```yaml
# .github/workflows/deploy.yml
name: Multi-Environment Deploy

on:
  push:
    branches: [main, staging, develop]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Determine environment
        id: env
        run: |
          if [[ $GITHUB_REF == 'refs/heads/main' ]]; then
            echo "environment=production" >> $GITHUB_OUTPUT
          elif [[ $GITHUB_REF == 'refs/heads/staging' ]]; then
            echo "environment=staging" >> $GITHUB_OUTPUT
          else
            echo "environment=development" >> $GITHUB_OUTPUT
          fi
      
      - name: Deploy to ${{ steps.env.outputs.environment }}
        run: |
          kubectl apply -f k8s/${{ steps.env.outputs.environment }}/
```

### Model Performance Testing

```yaml
- name: Test model performance
  run: |
    python -c "
    import torch
    import time
    from src.models import create_model
    
    model = create_model('cnn')
    model.eval()
    
    # Measure inference time
    dummy_input = torch.randn(1, 3, 224, 224)
    
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    # Assert performance requirement
    assert avg_time < 0.05, f'Inference too slow: {avg_time}s'
    "
```

---

## Summary

This deep dive covered:

✅ **Architecture internals** - How CNN and ViT actually work  
✅ **PyTorch Lightning patterns** - Best practices for training  
✅ **Advanced preprocessing** - Production-grade augmentation  
✅ **MLflow mastery** - Experiment tracking and model registry  
✅ **FastAPI optimization** - High-performance API patterns  
✅ **Kubernetes advanced** - Production-ready orchestration  
✅ **Performance tuning** - Quantization, ONNX, batching  
✅ **Monitoring** - Custom metrics and drift detection  
✅ **Production patterns** - Graceful shutdown, health checks  
✅ **Custom development** - Building new architectures  
✅ **CI/CD automation** - Multi-environment deployment  

**Next level:** Edge deployment, model explainability, federated learning!
