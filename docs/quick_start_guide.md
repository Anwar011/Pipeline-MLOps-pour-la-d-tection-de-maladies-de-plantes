# 🚀 Quick Start Guide - Plant Disease Detection MLOps

## ⚡ 5-Minute Setup

### Prerequisites
```bash
# Required
- Python 3.9+
- Docker & Docker Compose
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA 11.8
```

## Option 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone <repo-url>
cd Pipeline-MLOps-pour-la-d-tection-de-maladies-de-plantes

# 2. Start all services
docker-compose -f docker/docker-compose.yml up -d

# 3. Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090

# 4. Test API
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/plant_image.jpg"
```

## Option 2: Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (if training)
# https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to: data/raw/PlantVillage/

# 3. Train a model (optional)
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model cnn

# 4. Run API
python scripts/run_api.py

# 5. Test
curl http://localhost:8000/health
```

## Option 3: Kubernetes

```bash
# 1. Build and push image
docker build -f docker/Dockerfile -t your-registry/plant-disease-mlops:latest .
docker push your-registry/plant-disease-mlops:latest

# 2. Deploy to K8s
kubectl apply -f k8s/

# 3. Check status
kubectl get all -n mlops

# 4. Port forward to test
kubectl port-forward svc/plant-disease-service 8000:80 -n mlops
```

## 📊 Common Tasks

### Train a New Model
```bash
# CNN (ResNet50)
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model cnn

# Vision Transformer
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model vit
```

### Make Predictions
```python
import requests

# Single image
with open('leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
print(response.json())

# Output:
# {
#   "prediction": "Tomato_Early_blight",
#   "confidence": 0.94,
#   "probabilities": {...}
# }
```

### View Experiments
```bash
# Access MLflow UI
open http://localhost:5000

# Compare models
# View metrics, parameters
# Download trained models
```

### Monitor Performance
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Grafana dashboards
open http://localhost:3000
# Username: admin
# Password: admin
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Adjust training epochs
training:
  epochs: 50  # Change from 5 to 50 for full training

# Change model architecture
model:
  architecture: "efficientnet"  # or "resnet50", "vgg16"

# Adjust batch size
data:
  batch_size: 64  # Increase if you have more RAM
```

## 🐛 Troubleshooting

**API won't start:**
```bash
# Check if model exists
ls -lh models/production/model.ckpt

# If not, create dummy model
python create_dummy_model.py
```

**Out of memory:**
```yaml
# Reduce batch_size in config.yaml
data:
  batch_size: 16  # Lower value
```

**Docker issues:**
```bash
# Rebuild without cache
docker-compose -f docker/docker-compose.yml build --no-cache

# Check logs
docker-compose -f docker/docker-compose.yml logs -f plant-disease-api
```

## 📚 Supported Plant Diseases

Currently supports 15 classes:
- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: 9 different conditions including healthy

See `data/class_mapping.json` for full list.

## 🎯 Next Steps

1. ✅ Train your first model
2. ✅ Make predictions via API
3. ✅ View experiments in MLflow
4. ✅ Set up monitoring in Grafana
5. ✅ Deploy to Kubernetes (production)

## 📞 Support

- Documentation: See `comprehensive_analysis.md`
- Issues: GitHub Issues
- Config help: Check `config.yaml` comments
