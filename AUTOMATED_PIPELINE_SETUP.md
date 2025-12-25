# 🚀 Pipeline Automatisé MLOps - Configuration Complète

## ✅ Ce qui a été créé

J'ai mis en place un pipeline automatisé complet qui:

1. **Détecte automatiquement les changements DVC** (nouvelles données)
2. **Exécute le pipeline DVC** (prepare_data → train → evaluate → export)
3. **Enregistre les données et modèles dans MLflow**
4. **Reconstruit l'image Docker** avec le nouveau modèle
5. **Redéploie localement** avec Docker Compose

## 📁 Fichiers créés

### Scripts principaux

1. **`scripts/monitor_dvc_changes.py`**
   - Surveille les changements dans `dvc.lock` et fichiers `.dvc`
   - Détecte quand de nouvelles données sont ajoutées
   - Peut fonctionner en mode surveillance continue

2. **`scripts/run_automated_pipeline.py`**
   - Script principal qui orchestre tout le pipeline
   - Exécute toutes les étapes automatiquement
   - Gère les erreurs et affiche un résumé

3. **`scripts/watch_and_trigger.py`**
   - Surveillance continue avec déclenchement automatique
   - Vérifie les changements toutes les 30 secondes (configurable)
   - Déclenche le pipeline dès qu'un changement est détecté

4. **`scripts/quick_start.sh`** et **`scripts/quick_start.ps1`**
   - Scripts de démarrage rapide pour Linux/Mac et Windows
   - Vérifient les prérequis
   - Interface simple pour choisir l'option

### Documentation

- **`scripts/AUTOMATED_PIPELINE_README.md`** - Guide complet d'utilisation

### Modifications

- **`docker/docker-compose.yml`** - Service API ajouté et configuré
- **`src/train.py`** - Amélioration de l'enregistrement des données dans MLflow

## 🚀 Utilisation rapide

### Option 1: Exécution manuelle unique

```bash
# Exécuter le pipeline une fois
python scripts/run_automated_pipeline.py

# Forcer même sans changements
python scripts/run_automated_pipeline.py --force
```

### Option 2: Surveillance continue (recommandé)

```bash
# Démarrer la surveillance (déclenchement automatique)
python scripts/watch_and_trigger.py

# Avec intervalle personnalisé (60 secondes)
python scripts/watch_and_trigger.py --interval 60
```

### Option 3: Script de démarrage rapide

**Linux/Mac:**
```bash
bash scripts/quick_start.sh
```

**Windows:**
```powershell
.\scripts\quick_start.ps1
```

## 📊 Flux de travail

```
1. Ajouter nouvelles données
   ↓
   dvc add data/raw/PlantVillage/NewClass
   git add data/raw/PlantVillage/NewClass.dvc dvc.lock
   git commit -m "Add new data"
   
2. Pipeline détecte automatiquement (si surveillance active)
   OU
   Exécuter manuellement: python scripts/run_automated_pipeline.py
   
3. Pipeline exécute:
   - dvc repro (prepare_data → train → evaluate → export)
   - Enregistrement dans MLflow
   - Construction Docker
   - Déploiement local
   
4. API disponible avec nouveau modèle
   http://localhost:8000
```

## 🔧 Configuration

Le pipeline utilise `config.yaml` pour:
- Chemins des données et modèles
- Configuration MLflow (tracking_uri, experiment_name)
- Configuration Docker (image_name, tag)
- Paramètres d'entraînement

## 📝 Exemple complet

### 1. Démarrer la surveillance

```bash
# Terminal 1
python scripts/watch_and_trigger.py
```

### 2. Ajouter de nouvelles données

```bash
# Terminal 2
# Ajouter de nouvelles images dans data/raw/PlantVillage/NewClass/

# Ajouter à DVC
dvc add data/raw/PlantVillage/NewClass

# Commit
git add data/raw/PlantVillage/NewClass.dvc dvc.lock
git commit -m "Add new plant disease class"
```

### 3. Le pipeline se déclenche automatiquement!

Dans le Terminal 1, vous verrez:
```
🔍 Vérification #1...
🔍 Changements DVC détectés!
🔄 CHANGEMENTS DÉTECTÉS - DÉCLENCHEMENT DU PIPELINE
🚀 Exécution du pipeline DVC...
✅ Pipeline DVC exécuté avec succès
📝 Vérification de l'enregistrement MLflow...
✅ Modèle de production trouvé
🐳 Reconstruction de l'image Docker...
✅ Image Docker construite avec succès
🚀 Déploiement local avec Docker Compose...
✅ Services déployés avec succès
🎉 PIPELINE TERMINÉ AVEC SUCCÈS!
```

### 4. Tester la nouvelle API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"
```

## 🎯 Services disponibles après déploiement

- **API**: http://localhost:8000
  - `/docs` - Documentation Swagger
  - `/health` - Health check
  - `/predict` - Prédiction
  - `/metrics` - Métriques Prometheus

- **MLflow**: http://localhost:5000
  - Expériences et runs
  - Model Registry
  - Métriques et graphiques

- **Grafana**: http://localhost:3000 (admin/admin)
  - Dashboards de monitoring

- **Prometheus**: http://localhost:9091
  - Métriques brutes

## 🐛 Dépannage

### Le pipeline ne détecte pas les changements

```bash
# Vérifier manuellement
python scripts/monitor_dvc_changes.py

# Forcer l'exécution
python scripts/run_automated_pipeline.py --force
```

### Erreur DVC

```bash
# Vérifier le statut
dvc status

# Exécuter manuellement
dvc repro
```

### Erreur Docker

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

# Redémarrer
docker-compose -f docker/docker-compose.yml restart plant-disease-api
```

## 📚 Documentation complète

Consultez `scripts/AUTOMATED_PIPELINE_README.md` pour:
- Guide détaillé de chaque étape
- Options avancées
- Configuration personnalisée
- Workflows complexes

## ✅ Checklist de vérification

Avant d'utiliser le pipeline:

- [ ] DVC installé et initialisé (`dvc init`)
- [ ] Docker et Docker Compose installés
- [ ] MLflow accessible (démarre avec docker-compose)
- [ ] Données dans `data/raw/PlantVillage/`
- [ ] `config.yaml` configuré correctement
- [ ] Dépendances Python installées (`pip install -r requirements.txt`)

## 🎉 Prêt à utiliser!

Le pipeline est maintenant complètement automatisé. Il suffit de:

1. **Démarrer la surveillance**: `python scripts/watch_and_trigger.py`
2. **Ajouter de nouvelles données** avec DVC
3. **Le pipeline se déclenche automatiquement!**

Tout est enregistré dans MLflow et l'API est redéployée avec le nouveau modèle.

---

**💡 Astuce**: Pour un premier test, utilisez `--force` pour exécuter le pipeline même sans changements DVC.

