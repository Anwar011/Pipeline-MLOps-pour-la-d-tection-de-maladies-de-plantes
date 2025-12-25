# 🚀 Configuration GitHub Actions pour le Pipeline Automatisé

## 📋 Vue d'ensemble

Le workflow GitHub Actions `mlops-automated-pipeline.yml` automatise complètement votre pipeline MLOps:

1. **Détecte automatiquement** les changements DVC (quand `dvc.lock` change)
2. **Exécute le pipeline DVC** complet (prepare_data → train → evaluate → export)
3. **Enregistre dans MLflow** (métriques, modèles, données)
4. **Construit l'image Docker** avec le nouveau modèle
5. **Push vers Docker Hub** (prêt pour déploiement local)

## 🔧 Configuration requise

### 1. Secrets GitHub

Allez dans **Settings → Secrets and variables → Actions** et ajoutez:

#### Secrets obligatoires (pour Docker Hub)

- `DOCKER_USERNAME` - Votre nom d'utilisateur Docker Hub
- `DOCKER_PASSWORD` - Votre token Docker Hub (pas votre mot de passe!)

#### Secrets optionnels

- `MLFLOW_TRACKING_URI` - URI du serveur MLflow (ex: `http://mlflow.example.com:5000`)
  - Si non défini, utilise `file:./mlruns` (local dans le runner)
  
- `DVC_REMOTE_URL` - URL du remote DVC (S3, Google Drive, etc.)
- `DVC_ACCESS_KEY_ID` - Clé d'accès pour le remote DVC
- `DVC_SECRET_ACCESS_KEY` - Clé secrète pour le remote DVC

### 2. Créer un token Docker Hub

1. Allez sur https://hub.docker.com/settings/security
2. Cliquez sur "New Access Token"
3. Donnez un nom (ex: "github-actions")
4. Copiez le token et ajoutez-le comme secret `DOCKER_PASSWORD`

### 3. Configurer DVC Remote (optionnel)

Si vos données sont sur un remote DVC (S3, Google Drive, etc.):

```bash
# Exemple avec S3
dvc remote add -d storage s3://my-bucket/plant-disease-data
dvc remote modify storage access_key_id YOUR_ACCESS_KEY
dvc remote modify storage secret_access_key YOUR_SECRET_KEY

# Commit la configuration
git add .dvc/config
git commit -m "Configure DVC remote"
```

## 🚀 Utilisation

### Déclenchement automatique

Le workflow se déclenche automatiquement quand:

1. **`dvc.lock` change** - Nouvelles données ajoutées
2. **`dvc.yaml` change** - Pipeline DVC modifié
3. **Fichiers `.dvc` changent** - Données trackées
4. **Code d'entraînement change** - `src/train.py`, `src/models.py`, etc.

### Déclenchement manuel

1. Allez dans **Actions** sur GitHub
2. Sélectionnez **"🤖 MLOps Automated Pipeline"**
3. Cliquez sur **"Run workflow"**
4. Options disponibles:
   - **Force training**: Forcer l'entraînement même sans changements
   - **Model type**: Choisir `cnn` ou `vit`
   - **Skip deploy**: Ignorer la construction Docker

### Workflow d'utilisation typique

```bash
# 1. Ajouter de nouvelles données
dvc add data/raw/PlantVillage/NewClass

# 2. Commit les changements
git add data/raw/PlantVillage/NewClass.dvc dvc.lock
git commit -m "Add new plant disease class data"
git push

# 3. Le workflow GitHub Actions se déclenche automatiquement!
# Vérifier dans l'onglet "Actions" de votre repository
```

## 📊 Étapes du workflow

### Job 1: 🔍 Check DVC Changes
- Vérifie si `dvc.lock` ou le code a changé
- Détermine si l'entraînement est nécessaire

### Job 2: 📦 Pull Data (DVC)
- Pull les données depuis le remote DVC
- Préparation des données pour l'entraînement

### Job 3: 🔄 Run DVC Pipeline
- Exécute `dvc repro` (pipeline complet)
- Ou exécute manuellement: prepare_data → train → evaluate → export
- Enregistre automatiquement dans MLflow

### Job 4: 🐳 Build Docker Image
- Trouve le meilleur modèle dans `models/checkpoints/`
- Copie vers `models/production/model.ckpt`
- Construit l'image Docker avec le nouveau modèle
- Push vers Docker Hub (si credentials configurés)

### Job 5: 📊 Summary
- Génère un résumé dans GitHub Actions
- Affiche les instructions pour déployer localement

## 🐳 Déploiement local après le workflow

Une fois le workflow terminé:

### Option 1: Utiliser l'image Docker Hub

```bash
# Pull l'image avec la nouvelle version
docker pull YOUR_USERNAME/plant-disease-mlops:v20241208-123456-abc1234

# Mettre à jour docker-compose.yml
# Changez l'image dans docker/docker-compose.yml:
# image: YOUR_USERNAME/plant-disease-mlops:v20241208-123456-abc1234

# Redémarrer les services
docker-compose -f docker/docker-compose.yml up -d
```

### Option 2: Utiliser les artifacts

Le workflow sauvegarde les artifacts:
- `trained-model`: Modèle entraîné, métriques, plots
- `training-data`: Données utilisées

Vous pouvez les télécharger depuis l'interface GitHub Actions.

## 🔍 Monitoring

### Vérifier le statut

1. Allez dans **Actions** sur GitHub
2. Cliquez sur le dernier workflow run
3. Vérifiez chaque job pour voir les logs

### Logs importants

- **Check Changes**: Affiche quels fichiers ont changé
- **DVC Pipeline**: Affiche les métriques d'entraînement
- **Docker Build**: Affiche l'image créée et les tags

### MLflow

Si vous avez configuré `MLFLOW_TRACKING_URI`:
- Accédez à l'URI pour voir les runs
- Tous les paramètres, métriques et modèles sont enregistrés

## 🐛 Dépannage

### Le workflow ne se déclenche pas

**Problème**: Les changements DVC ne sont pas détectés

**Solution**:
1. Vérifiez que `dvc.lock` est commité
2. Vérifiez que les fichiers `.dvc` sont commités
3. Utilisez "Run workflow" manuellement avec "Force training"

### Erreur DVC pull

**Problème**: `dvc pull` échoue

**Solution**:
1. Vérifiez que les secrets DVC sont configurés
2. Vérifiez que le remote DVC est correctement configuré
3. Le workflow utilisera les données locales si disponibles

### Erreur Docker build

**Problème**: L'image Docker ne se construit pas

**Solution**:
1. Vérifiez que `models/production/model.ckpt` existe
2. Vérifiez les logs du job "Build Docker Image"
3. Le workflow crée un dummy model si nécessaire

### Erreur Docker push

**Problème**: L'image ne se push pas vers Docker Hub

**Solution**:
1. Vérifiez que `DOCKER_USERNAME` et `DOCKER_PASSWORD` sont configurés
2. Vérifiez que le token Docker Hub est valide
3. L'image est construite localement même si le push échoue

## 📝 Exemple de workflow complet

```yaml
# .github/workflows/mlops-automated-pipeline.yml
# Déjà créé et configuré!
```

## ✅ Checklist de configuration

- [ ] Secrets GitHub configurés (`DOCKER_USERNAME`, `DOCKER_PASSWORD`)
- [ ] Token Docker Hub créé et ajouté comme secret
- [ ] DVC remote configuré (si données distantes)
- [ ] Secrets DVC configurés (si nécessaire)
- [ ] MLflow tracking URI configuré (optionnel)
- [ ] Workflow testé avec "Run workflow" manuel

## 🎯 Prochaines étapes

Une fois le workflow configuré:

1. **Testez manuellement** avec "Run workflow"
2. **Ajoutez de nouvelles données** avec DVC
3. **Push vers GitHub** - Le workflow se déclenche automatiquement!
4. **Vérifiez les résultats** dans l'onglet Actions
5. **Déployez localement** avec la nouvelle image Docker

## 📚 Ressources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [DVC Documentation](https://dvc.org/doc)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**💡 Astuce**: Pour tester sans push, utilisez "Run workflow" manuellement avec "Force training" activé.

