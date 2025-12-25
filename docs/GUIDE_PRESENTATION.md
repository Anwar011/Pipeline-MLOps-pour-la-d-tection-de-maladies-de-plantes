# 🎓 Guide de Présentation - Pipeline MLOps
## Détection de Maladies de Plantes

> **Document destiné à la soutenance PFA**  
> Préparé pour démonstration devant l'encadrant

---

## 📋 Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture MLOps](#2-architecture-mlops)
3. [Démonstration pas à pas](#3-démonstration-pas-à-pas)
4. [Points clés à présenter](#4-points-clés-à-présenter)
5. [Commandes de démonstration](#5-commandes-de-démonstration)
6. [FAQ pour la soutenance](#6-faq-pour-la-soutenance)

---

## 1. Vue d'ensemble du projet

### 🎯 Problématique
> "Comment concevoir un pipeline MLOps capable d'entraîner, déployer et surveiller automatiquement un modèle de détection de maladies végétales, tout en assurant la reproductibilité et la scalabilité du système ?"

### ✅ Solution implémentée

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS COMPLET                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 DataOps          🤖 ModelOps          🚀 DeploymentOps          │
│  ─────────          ──────────          ──────────────             │
│  • DVC               • PyTorch Lightning • Docker                   │
│  • Augmentation      • MLflow Tracking   • Kubernetes               │
│  • Versioning        • Model Registry    • GitHub Actions           │
│                      • ONNX Export       • Prometheus/Grafana       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture MLOps

### 📐 Diagramme d'Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dataset    │────▶│     DVC      │────▶│  Processed   │
│ PlantVillage │     │  Versioning  │     │    Data      │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   MLflow     │◀────│   Training   │◀────│   PyTorch    │
│   Tracking   │     │   Pipeline   │     │  Lightning   │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Model     │────▶│    Docker    │────▶│  Kubernetes  │
│   Registry   │     │    Image     │     │  Deployment  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│  Prometheus  │────▶│   Grafana    │
│     API      │     │   Metrics    │     │  Dashboard   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 📁 Structure du Projet

```
Pipeline-MLOps/
├── 📂 src/                    # Code source principal
│   ├── api.py                 # API FastAPI avec Prometheus
│   ├── train.py               # Script d'entraînement + MLflow
│   ├── models.py              # CNN & ViT avec PyTorch Lightning
│   └── data_preprocessing.py  # Augmentation avec Albumentations
│
├── 📂 scripts/                # Scripts MLOps
│   ├── prepare_data.py        # Préparation données (DVC)
│   ├── evaluate.py            # Évaluation (F1, ROC, Confusion)
│   ├── export_model.py        # Export ONNX + Registry
│   └── drift_analysis.py      # Détection drift (Evidently)
│
├── 📂 docker/                 # Conteneurisation
│   ├── Dockerfile.inference   # Image optimisée production
│   └── docker-compose.yml     # Stack complète locale
│
├── 📂 k8s/                    # Orchestration Kubernetes
│   ├── deployment.yaml        # Déploiement avec replicas
│   ├── service.yaml           # LoadBalancer
│   └── hpa.yaml              # Auto-scaling
│
├── 📂 .github/workflows/      # CI/CD
│   └── mlops-pipeline.yml     # Pipeline complet
│
├── 📂 monitoring/             # Observabilité
│   ├── prometheus.yml         # Config Prometheus
│   └── grafana/              # Dashboards
│
├── 📄 dvc.yaml                # Pipeline DVC
├── 📄 config.yaml             # Configuration centralisée
└── 📄 requirements.txt        # Dépendances
```

---

## 3. Démonstration pas à pas

### 🔧 Prérequis (à installer avant la démo)

```powershell
# 1. Installer Python 3.10+ depuis python.org
# 2. Démarrer Docker Desktop
# 3. Installer les dépendances
pip install -r requirements.txt
```

### 📊 Démo 1: Pipeline DVC (DataOps)

```powershell
# Montrer le fichier dvc.yaml
cat dvc.yaml

# Exécuter la préparation des données
python scripts/prepare_data.py --config config.yaml

# Visualiser le pipeline
dvc dag
```

**Points à expliquer:**
- Versioning des données avec DVC
- Reproductibilité des expériences
- Division 70/20/10 (train/val/test)

### 🤖 Démo 2: Entraînement avec MLflow (ModelOps)

```powershell
# Démarrer MLflow UI (dans un terminal séparé)
mlflow ui --port 5000

# Lancer l'entraînement
python src/train.py --model cnn --dataset data/raw/PlantVillage --config config.yaml
```

**Points à expliquer:**
- Tracking automatique des métriques
- Logging des hyperparamètres
- Model Registry pour versioning

### 🚀 Démo 3: API FastAPI (DeploymentOps)

```powershell
# Lancer l'API localement
python src/api.py

# Dans un autre terminal, tester l'API
curl http://localhost:8000/health
curl http://localhost:8000/classes
```

**Points à expliquer:**
- Endpoint /predict pour l'inférence
- Métriques Prometheus automatiques
- Temps de réponse < 2s

### 🐳 Démo 4: Docker & Kubernetes

```powershell
# Build de l'image Docker
docker build -f docker/Dockerfile.inference -t plant-disease-api .

# Lancer avec Docker
docker run -p 8000:8000 plant-disease-api

# Voir les manifests Kubernetes
cat k8s/deployment.yaml
cat k8s/hpa.yaml
```

**Points à expliquer:**
- Image optimisée pour production
- Auto-scaling avec HPA
- Health checks et readiness probes

### 📈 Démo 5: Monitoring

```powershell
# Lancer la stack de monitoring
docker-compose -f docker/docker-compose.yml up -d

# Accéder aux interfaces
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9091
# MLflow: http://localhost:5000
```

---

## 4. Points clés à présenter

### ✅ Conformité au Cahier des Charges

| Exigence | Implémentation | Fichier |
|----------|---------------|---------|
| DVC - Gestion données | ✅ Pipeline 5 stages | `dvc.yaml` |
| MLflow - Tracking | ✅ Métriques + Registry | `src/train.py` |
| CNN/ViT | ✅ ResNet50, EfficientNet | `src/models.py` |
| FastAPI | ✅ /predict, /health | `src/api.py` |
| Docker | ✅ Multi-stage build | `docker/Dockerfile.inference` |
| Kubernetes | ✅ Deployment + HPA | `k8s/` |
| CI/CD | ✅ GitHub Actions | `.github/workflows/` |
| Prometheus | ✅ Instrumentator | `src/api.py` |
| Grafana | ✅ Dashboard custom | `monitoring/grafana/` |
| Evidently | ✅ Drift detection | `scripts/drift_analysis.py` |
| Tests | ✅ Unit + API | `tests/` |

### 🎯 Métriques Attendues

- **Accuracy modèle**: 90-95%
- **F1-Score**: > 0.90
- **Temps inférence**: < 100ms
- **Temps réponse API**: < 2s
- **Disponibilité**: 99.9% (avec replicas K8s)

---

## 5. Commandes de démonstration

### 🚀 Scripts Bash disponibles

Tous les scripts sont dans le dossier `scripts/` et utilisent des chemins relatifs.

#### Script principal (menu interactif)
```bash
cd scripts
./main.sh                    # Afficher le menu complet
./main.sh pipeline          # Lancer le pipeline complet
./main.sh api               # Lancer l'API
./main.sh monitoring        # Démarrer le monitoring
./main.sh tests             # Lancer tous les tests
./main.sh demo              # Présentation interactive
```

#### Pipeline MLOps complet
```bash
cd scripts

# Vérifier les prérequis
./run_pipeline.sh check

# Installer les dépendances
./run_pipeline.sh install

# Initialiser DVC
./run_pipeline.sh init

# Exécuter le pipeline complet (5 étapes)
./run_pipeline.sh pipeline

# Lancer les tests
./run_pipeline.sh test

# Construire les images Docker
./run_pipeline.sh docker

# Tout faire automatiquement
./run_pipeline.sh all
```

#### API FastAPI
```bash
cd scripts

# Mode développement (avec rechargement auto)
./run_api.sh dev

# Mode production (recommandé)
./run_api.sh gunicorn

# Tester l'API automatiquement
./run_api.sh test
```

#### Monitoring Prometheus + Grafana
```bash
cd scripts

# Démarrer la stack de monitoring
./run_monitoring.sh start

# Vérifier le statut
./run_monitoring.sh status

# Tester les services
./run_monitoring.sh test

# Afficher les logs
./run_monitoring.sh logs
```

#### Tests automatisés
```bash
cd scripts

# Tests unitaires avec couverture
./run_tests.sh unit

# Tests d'intégration API
./run_tests.sh api

# Tests de performance
./run_tests.sh perf

# Tests de sécurité
./run_tests.sh security

# Tests du pipeline DVC
./run_tests.sh pipeline

# Tous les tests
./run_tests.sh all

# Générer un rapport
./run_tests.sh report
```

#### Déploiement Kubernetes
```bash
cd scripts

# Vérifier les prérequis
./deploy_k8s.sh check

# Construire et pousser les images
./deploy_k8s.sh build

# Déploiement complet
./deploy_k8s.sh deploy

# Tester le déploiement
./deploy_k8s.sh test

# Afficher le statut
./deploy_k8s.sh status
```

#### Démonstration interactive
```bash
cd scripts

# Présentation complète du projet
./demo_presentation.sh
```

### 📋 URLs importantes

Après avoir lancé les services :

- **API FastAPI**: http://localhost:8000
- **Documentation API**: http://localhost:8000/docs
- **Métriques Prometheus**: http://localhost:8000/metrics
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **MLflow**: http://localhost:5000

### 🎯 Workflow de démonstration recommandé

```bash
# 1. Vérification du projet
cd scripts && ./run_pipeline.sh check

# 2. Pipeline complet
./run_pipeline.sh all

# 3. Lancement de l'API
./run_api.sh gunicorn &

# 4. Démarrage du monitoring
./run_monitoring.sh start

# 5. Tests complets
./run_tests.sh all

# 6. Présentation finale
./demo_presentation.sh
```

---

## 6. FAQ pour la soutenance

### Q1: "Pourquoi DVC plutôt que Git LFS?"
> DVC permet de créer des pipelines reproductibles avec des dépendances explicites entre les étapes. Il gère aussi le versioning des données volumineuses avec des backends cloud (S3, GCS).

### Q2: "Comment gérez-vous le drift des données?"
> Nous utilisons Evidently AI pour détecter automatiquement les dérives de distribution entre les données d'entraînement et de production. Le script `drift_analysis.py` génère des rapports HTML.

### Q3: "Quelle est la stratégie de déploiement?"
> Déploiement blue-green via Kubernetes avec:
> - Rolling updates pour zéro downtime
> - HPA pour auto-scaling (2-10 replicas)
> - Health checks pour haute disponibilité

### Q4: "Comment assurez-vous la reproductibilité?"
> - DVC pour le versioning des données
> - MLflow pour le tracking des expériences
> - Docker pour l'environnement
> - Config centralisée dans `config.yaml`

### Q5: "Quelles métriques surveillez-vous en production?"
> - Latence API (p50, p95, p99)
> - Taux de requêtes (QPS)
> - Distribution de confiance des prédictions
> - CPU/Mémoire des pods Kubernetes

---

## 📊 Slides suggérées pour la présentation

1. **Introduction** (2 min)
   - Contexte agriculture + IA
   - Problématique

2. **État de l'art** (3 min)
   - CNN vs ViT
   - Outils MLOps

3. **Architecture** (5 min)
   - Diagramme pipeline
   - Choix technologiques

4. **Démonstration** (10 min)
   - DVC pipeline
   - MLflow tracking
   - API FastAPI
   - Monitoring

5. **Résultats** (3 min)
   - Métriques modèle
   - Performance API

6. **Conclusion** (2 min)
   - Objectifs atteints
   - Perspectives

---

## 🚦 Checklist avant la démo

- [ ] Python 3.10+ installé et dans le PATH
- [ ] Docker Desktop démarré
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Dataset téléchargé dans `data/raw/`
- [ ] Modèle entraîné dans `models/checkpoints/`
- [ ] MLflow UI accessible (port 5000)
- [ ] Grafana/Prometheus up (docker-compose)

---

**Bonne soutenance! 🎓**
