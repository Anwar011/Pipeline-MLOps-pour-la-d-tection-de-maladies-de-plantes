# 🚀 Scripts Bash - Pipeline MLOps
## Détection de Maladies de Plantes

Ce dossier contient tous les scripts Bash pour gérer le pipeline MLOps complet.

## 📋 Scripts disponibles

### `main.sh` - Script principal (menu interactif)
```bash
./main.sh              # Afficher le menu complet
./main.sh pipeline     # Pipeline complet
./main.sh api          # Lancer l'API
./main.sh monitoring   # Monitoring
./main.sh tests        # Tests
./main.sh demo         # Démonstration
```

### `run_pipeline.sh` - Pipeline MLOps complet
```bash
./run_pipeline.sh check     # Vérifier prérequis
./run_pipeline.sh install   # Installer dépendances
./run_pipeline.sh init      # Initialiser DVC
./run_pipeline.sh pipeline  # Exécuter pipeline (5 étapes)
./run_pipeline.sh test      # Tests unitaires
./run_pipeline.sh docker    # Construire images
./run_pipeline.sh all       # Tout automatisé
```

### `run_api.sh` - API FastAPI
```bash
./run_api.sh dev       # Développement (rechargement auto)
./run_api.sh prod      # Production (uvicorn)
./run_api.sh gunicorn  # Production (recommandé)
./run_api.sh test      # Tests automatiques
```

### `run_monitoring.sh` - Prometheus + Grafana
```bash
./run_monitoring.sh start    # Démarrer stack
./run_monitoring.sh stop     # Arrêter
./run_monitoring.sh status   # Statut et URLs
./run_monitoring.sh test     # Tester services
./run_monitoring.sh logs     # Logs
```

### `run_tests.sh` - Tests automatisés
```bash
./run_tests.sh unit      # Tests unitaires
./run_tests.sh api       # Tests API
./run_tests.sh perf      # Performance
./run_tests.sh security  # Sécurité
./run_tests.sh pipeline  # Pipeline DVC
./run_tests.sh all       # Tous les tests
```

### `deploy_k8s.sh` - Déploiement Kubernetes
```bash
./deploy_k8s.sh check    # Prérequis
./deploy_k8s.sh build    # Images
./deploy_k8s.sh deploy   # Déploiement
./deploy_k8s.sh test     # Tests
./deploy_k8s.sh status   # Statut
```

### `demo_presentation.sh` - Présentation interactive
```bash
./demo_presentation.sh   # Démo complète
```

## 🎯 Workflow rapide

```bash
# Depuis le dossier scripts/
./run_pipeline.sh all        # Pipeline complet
./run_api.sh gunicorn &      # API en production
./run_monitoring.sh start    # Monitoring
./run_tests.sh all           # Tests
./demo_presentation.sh       # Présentation
```

## 📊 URLs importantes

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **MLflow**: http://localhost:5000

## ✅ Conformité cahier des charges

- ✅ DVC - Gestion données
- ✅ MLflow - Tracking
- ✅ PyTorch Lightning
- ✅ FastAPI - API REST
- ✅ Docker - Conteneurisation
- ✅ Kubernetes - Orchestration
- ✅ GitHub Actions - CI/CD
- ✅ Prometheus - Métriques
- ✅ Grafana - Dashboard
- ✅ Evidently - Drift Detection
- ✅ Tests unitaires
- ✅ Export ONNX
- ✅ Temps réponse < 2s

---

**Pour la soutenance**: `./demo_presentation.sh`