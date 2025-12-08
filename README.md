# Pipeline MLOps pour la Détection de Maladies de Plantes.

[![CI](https://github.com/your-username/plant-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/plant-disease-mlops/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)

Un pipeline MLOps complet pour la détection automatique de maladies de plantes utilisant Deep Learning, MLflow, Docker et Kubernetes.

## 🎯 Vue d'ensemble

Ce projet implémente un pipeline MLOps de bout en bout pour la classification automatique de maladies de plantes à partir d'images. Il comprend :

- **Prétraitement et augmentation des données** avec Albumentations
- **Modèles CNN et Vision Transformer** avec PyTorch Lightning
- **Suivi d'expériences** avec MLflow
- **API d'inférence** avec FastAPI et Prometheus
- **Conteneurisation** avec Docker
- **Déploiement automatisé** avec Kubernetes et GitHub Actions
- **Monitoring** avec Grafana et Prometheus

## 📚 Documentation Complète

Une documentation exhaustive est disponible dans le répertoire [`docs/`](docs/):

- **[🚀 Quick Start Guide](docs/quick_start_guide.md)** - Démarrage en 5 minutes (Docker, local, Kubernetes)
- **[🎯 Repository Walkthrough](docs/walkthrough.md)** - Parcours guidé de tout le système
- **[🌱 Comprehensive Analysis](docs/comprehensive_analysis.md)** - Référence technique complète (architecture, déploiement, monitoring)
- **[🔬 Technical Deep Dive](docs/technical_deep_dive.md)** - Sujets avancés et optimisations

👉 **Nouveau dans le projet ?** Commencez par le [Quick Start Guide](docs/quick_start_guide.md)

## 📁 Structure du projet

```
plant-disease-mlops/
├── config.yaml              # Configuration principale
├── requirements.txt         # Dépendances Python
├── docker/
│   ├── Dockerfile.train    # Image pour l'entraînement (GPU)
│   ├── Dockerfile.inference # Image optimisée pour la prod (CPU)
│   └── docker-compose.yml  # Services locaux
├── k8s/                    # Configurations Kubernetes
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── storage.yaml
│   ├── namespace.yaml
│   └── hpa.yaml
├── monitoring/             # Configurations de monitoring
│   ├── prometheus.yml
│   └── grafana/
├── src/                    # Code source
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── api.py
├── scripts/                # Scripts utilitaires
│   ├── train_pipeline.py
│   └── run_api.py
├── models/                 # Modèles entraînés
├── data/                   # Données et mappings
├── notebooks/              # Notebooks Jupyter
└── .github/workflows/      # CI/CD GitHub Actions
```

## 🚀 Démarrage rapide

### Prérequis

- Python 3.9+
- Docker & Docker Compose
- kubectl (pour Kubernetes)
- Dataset PlantVillage (téléchargeable sur [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease))

### Installation

1. **Cloner le repository**
   ```bash
   git clone https://github.com/your-username/plant-disease-mlops.git
   cd plant-disease-mlops
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Télécharger le dataset**
   ```bash
   # Créer le répertoire des données
   mkdir -p data/raw

   # Télécharger PlantVillage dataset depuis Kaggle
   # Placer les images dans data/raw/PlantVillage/
   ```

### Utilisation locale avec Docker Compose

```bash
# Lancer tous les services (API, MLflow, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d

# Accéder aux services :
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Entraînement d'un modèle

```bash
# Entraîner un modèle CNN
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model cnn

# Entraîner un modèle Vision Transformer
python scripts/train_pipeline.py \
  --dataset data/raw/PlantVillage \
  --model vit
```

### Lancement de l'API

```bash
# Lancer l'API d'inférence
python scripts/run_api.py --host 0.0.0.0 --port 8000

# Ou avec Docker
docker run -p 8000:8000 anwar/plant-disease-mlops:latest
```

## 📊 API Documentation

### Endpoints principaux

- `GET /` - Informations générales
- `GET /health` - Vérification de santé
- `POST /predict` - Prédiction sur une image
- `POST /predict_batch` - Prédiction sur plusieurs images
- `GET /classes` - Liste des classes supportées
- `GET /metrics` - Métriques Prometheus

### Exemple d'utilisation

```python
import requests

# Prédiction sur une image
files = {'file': open('path/to/plant_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
result = response.json()

print(f"Maladie prédite: {result['prediction']}")
print(f"Confiance: {result['confidence']:.2f}")
```

## 🏗️ Architecture

### Pipeline MLOps

```mermaid
graph LR
    A[Dataset] --> B[Prétraitement]
    B --> C[Entraînement]
    C --> D[MLflow Registry]
    D --> E[Docker Image]
    E --> F[Kubernetes]
    F --> G[API FastAPI]
    G --> H[Monitoring]
```

### Modèles supportés

1. **CNN (Convolutional Neural Networks)**
   - ResNet50
   - EfficientNet-B0
   - VGG16

2. **Vision Transformer (ViT)**
   - ViT-Base (patch 16x16)

## 🔧 Configuration

Le fichier `config.yaml` contient tous les paramètres configurables :

```yaml
data:
  batch_size: 32
  image_size: [224, 224]
  train_split: 0.7

model:
  architecture: "resnet50"
  num_classes: 38
  pretrained: true

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
```

## 📈 Monitoring et Observabilité

### Métriques collectées

- **Performance API** : latence, taux de requêtes, erreurs
- **Prédictions** : distribution de confiance, classes prédites
- **Ressources** : CPU, mémoire, GPU
- **Modèle** : drift de données, dégradation de performance

### Tableaux de bord Grafana

- Métriques API temps réel
- Performance des prédictions
- Utilisation des ressources
- Statut des pods Kubernetes

## 🚢 Déploiement

### Développement

```bash
# Tests locaux
docker-compose -f docker/docker-compose.yml up

# Tests unitaires
pytest src/ -v --cov=src
```

### Production

```bash
# Build et push de l'image
docker build -f docker/Dockerfile -t your-registry/plant-disease-mlops:latest .
docker push your-registry/plant-disease-mlops:latest

# Déploiement Kubernetes
kubectl apply -f k8s/

# Vérification du déploiement
kubectl get pods -n mlops
kubectl logs -f deployment/plant-disease-api -n mlops
```

### CI/CD & Automation

Le projet inclut des workflows GitHub Actions avancés :

#### 1. Pipeline CI/CD (`ci.yml`)
- **Tests automatisés** sur chaque push/PR.
- **Build et push** de l'image Docker.
- **Déploiement** en staging (si branche `develop`).

#### 2. Entraînement Automatisé (`training.yml`)
Permet d'entraîner le modèle sur votre propre machine (Self-Hosted Runner) :
- **Déclenchement :** Automatique (push sur `data/`) ou Manuel.
- **Action :** Lance l'entraînement GPU dans un conteneur Docker.
- **Setup :** `Settings > Actions > Runners > New self-hosted runner`.

#### 3. Déploiement Azure (`deploy-azure.yml`)
Déploie l'API sur Azure Kubernetes Service (AKS) :
- **Déclenchement :** Push sur `main`.
- **Action :** Build image prod -> Push ACR -> Deploy AKS.
- **Requis :** Secrets Azure configurés.

## 📚 Développement

### Ajouter un nouveau modèle

1. Étendre la classe `PlantDiseaseModel` dans `src/models.py`
2. Ajouter la configuration dans `config.yaml`
3. Mettre à jour les transformations si nécessaire

### Tests

```bash
# Tests unitaires
pytest src/ -v

# Tests d'intégration
pytest tests/integration/ -v

# Linting
flake8 src/
black src/ --check
isort src/ --check
```

### Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Dataset PlantVillage (Kaggle)
- PyTorch Lightning team
- FastAPI community
- CNCF projects (Kubernetes, Prometheus)

## 📞 Support

Pour support ou questions :
- Ouvrir une issue sur GitHub
- Contacter : your-email@example.com

---

**Note**: Ce projet est développé dans le cadre d'un travail académique sur les pipelines MLOps pour l'agriculture intelligente.
