# 🚀 Pipeline Automatisé MLOps - Guide d'Utilisation

Ce guide explique comment utiliser le pipeline automatisé qui se déclenche lorsque de nouvelles données sont détectées par DVC.

## 📋 Vue d'ensemble

Le pipeline automatisé exécute les étapes suivantes:

1. **Détection des changements DVC** - Surveille `dvc.lock` et les fichiers `.dvc`
2. **Exécution du pipeline DVC** - Lance `dvc repro` pour préparer les données et entraîner
3. **Enregistrement MLflow** - Les données et modèles sont automatiquement enregistrés dans MLflow
4. **Construction Docker** - Reconstruit l'image Docker avec le nouveau modèle
5. **Déploiement local** - Redéploie l'API avec Docker Compose

## 🛠️ Prérequis

```bash
# Installer les dépendances
pip install -r requirements.txt

# Vérifier que DVC est installé
dvc --version

# Vérifier que Docker est installé
docker --version
docker-compose --version
```

## 🚀 Utilisation

### Option 1: Exécution manuelle unique

Exécuter le pipeline une fois:

```bash
# Depuis la racine du projet
python scripts/run_automated_pipeline.py
```

Options disponibles:
```bash
# Forcer l'exécution même sans changements DVC
python scripts/run_automated_pipeline.py --force

# Ignorer certaines étapes
python scripts/run_automated_pipeline.py --skip-dvc --skip-docker

# Utiliser un fichier de config différent
python scripts/run_automated_pipeline.py --config my_config.yaml
```

### Option 2: Surveillance continue (recommandé)

Démarrer la surveillance qui déclenche automatiquement le pipeline:

```bash
# Surveillance avec vérification toutes les 30 secondes (défaut)
python scripts/watch_and_trigger.py

# Personnaliser l'intervalle
python scripts/watch_and_trigger.py --interval 60
```

### Option 3: Vérification manuelle des changements

Vérifier si des changements DVC sont détectés:

```bash
# Vérification unique
python scripts/monitor_dvc_changes.py

# Mode surveillance continue
python scripts/monitor_dvc_changes.py --watch --interval 30
```

## 📊 Flux de travail typique

### 1. Ajouter de nouvelles données

```bash
# Ajouter de nouvelles images dans data/raw/PlantVillage/
# Par exemple, ajouter un nouveau dossier de classe

# Ajouter les données à DVC
dvc add data/raw/PlantVillage/NewClass

# Commit les changements
git add data/raw/PlantVillage/NewClass.dvc dvc.lock
git commit -m "Add new plant disease class data"
```

### 2. Déclencher le pipeline

**Automatique (si surveillance active):**
- Le pipeline se déclenche automatiquement dans les 30 secondes

**Manuel:**
```bash
python scripts/run_automated_pipeline.py
```

### 3. Vérifier les résultats

```bash
# Vérifier que le modèle est enregistré dans MLflow
# Ouvrir: http://localhost:5000

# Vérifier que l'API fonctionne
curl http://localhost:8000/health

# Vérifier les services Docker
docker-compose -f docker/docker-compose.yml ps
```

## 🔍 Détails des étapes

### Étape 1: Détection DVC

Le script `monitor_dvc_changes.py` vérifie:
- Modifications de `dvc.lock` (indique de nouvelles données)
- Modifications de `dvc.yaml` (indique un pipeline modifié)
- Modifications des fichiers `.dvc` dans `data/`

### Étape 2: Pipeline DVC

Exécute `dvc repro` qui lance:
- `prepare_data`: Préparation et division des données
- `train`: Entraînement du modèle
- `evaluate`: Évaluation du modèle
- `export_model`: Export vers `models/production/`

### Étape 3: MLflow

Le script `src/train.py` enregistre automatiquement:
- Paramètres d'entraînement
- Métriques (accuracy, loss, etc.)
- Artifacts (modèle, graphiques)
- Modèle dans le Model Registry

### Étape 4: Construction Docker

L'image Docker est reconstruite avec:
- Le nouveau modèle depuis `models/production/model.ckpt`
- Le code source mis à jour
- Les dépendances nécessaires

### Étape 5: Déploiement

Docker Compose:
- Arrête les services existants
- Reconstruit l'image API
- Redémarre tous les services (API, MLflow, Prometheus, Grafana)

## 📝 Configuration

Le pipeline utilise `config.yaml` pour:
- Chemins des données et modèles
- Configuration MLflow
- Configuration Docker
- Paramètres d'entraînement

## 🐛 Dépannage

### Le pipeline ne détecte pas les changements

```bash
# Vérifier manuellement
python scripts/monitor_dvc_changes.py

# Forcer l'exécution
python scripts/run_automated_pipeline.py --force
```

### Erreur lors de l'exécution DVC

```bash
# Vérifier que DVC est initialisé
dvc status

# Vérifier les dépendances
dvc dag

# Exécuter manuellement
dvc repro
```

### Erreur lors de la construction Docker

```bash
# Vérifier que le modèle existe
ls -lh models/production/model.ckpt

# Construire manuellement
docker build -f docker/Dockerfile.inference -t plant-disease-mlops:latest .
```

### L'API ne démarre pas

```bash
# Vérifier les logs
docker-compose -f docker/docker-compose.yml logs plant-disease-api

# Vérifier le health check
curl http://localhost:8000/health

# Redémarrer les services
docker-compose -f docker/docker-compose.yml restart
```

## 📊 Monitoring

Une fois le pipeline exécuté, vous pouvez accéder à:

- **API**: http://localhost:8000
  - Documentation: http://localhost:8000/docs
  - Health: http://localhost:8000/health
  - Métriques: http://localhost:8000/metrics

- **MLflow**: http://localhost:5000
  - Expériences et runs
  - Model Registry
  - Métriques et graphiques

- **Grafana**: http://localhost:3000
  - Login: admin/admin
  - Dashboards de monitoring

- **Prometheus**: http://localhost:9091
  - Métriques brutes
  - Requêtes PromQL

## 🔄 Workflow complet

```bash
# 1. Démarrer la surveillance (dans un terminal)
python scripts/watch_and_trigger.py

# 2. Dans un autre terminal, ajouter de nouvelles données
dvc add data/raw/PlantVillage/NewClass
git add data/raw/PlantVillage/NewClass.dvc dvc.lock
git commit -m "Add new data"

# 3. Le pipeline se déclenche automatiquement!
# Vérifier les logs dans le premier terminal

# 4. Tester la nouvelle API
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

## 📚 Scripts disponibles

| Script | Description |
|--------|-------------|
| `monitor_dvc_changes.py` | Surveille les changements DVC |
| `run_automated_pipeline.py` | Exécute le pipeline complet |
| `watch_and_trigger.py` | Surveillance continue + déclenchement automatique |

## ✅ Checklist

Avant d'utiliser le pipeline automatisé:

- [ ] DVC est installé et initialisé
- [ ] Docker et Docker Compose sont installés
- [ ] MLflow est accessible (http://localhost:5000)
- [ ] Les données sont dans `data/raw/`
- [ ] Le fichier `config.yaml` est configuré
- [ ] Les dépendances Python sont installées

## 🎯 Prochaines étapes

Une fois le pipeline fonctionnel:

1. Configurer un remote DVC (Google Drive, S3, etc.)
2. Ajouter des notifications (email, Slack) lors des entraînements
3. Configurer des seuils de qualité pour accepter/rejeter les modèles
4. Ajouter des tests automatiques avant le déploiement
5. Intégrer avec CI/CD (GitHub Actions, GitLab CI)

---

**💡 Astuce**: Pour un développement plus rapide, utilisez `--skip-docker` et `--skip-deploy` pour tester uniquement l'entraînement.

