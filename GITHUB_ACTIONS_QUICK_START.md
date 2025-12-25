# 🚀 GitHub Actions - Démarrage Rapide

## ✅ Ce qui a été créé

J'ai créé un workflow GitHub Actions complet (`mlops-automated-pipeline.yml`) qui automatise votre pipeline MLOps.

## 📋 Configuration rapide (5 minutes)

### 1. Configurer les secrets GitHub

Allez dans votre repository GitHub:
1. **Settings** → **Secrets and variables** → **Actions**
2. Cliquez sur **"New repository secret"**
3. Ajoutez ces secrets:

```
DOCKER_USERNAME = votre-username-dockerhub
DOCKER_PASSWORD = votre-token-dockerhub
```

**Comment créer un token Docker Hub:**
1. Allez sur https://hub.docker.com/settings/security
2. Cliquez sur **"New Access Token"**
3. Donnez un nom et copiez le token
4. Ajoutez-le comme secret `DOCKER_PASSWORD`

### 2. Tester le workflow

1. Allez dans l'onglet **Actions** de votre repository
2. Sélectionnez **"🤖 MLOps Automated Pipeline"**
3. Cliquez sur **"Run workflow"**
4. Cochez **"Force training"** pour tester
5. Cliquez sur **"Run workflow"**

## 🎯 Utilisation normale

### Déclenchement automatique

Le workflow se déclenche automatiquement quand vous:

```bash
# 1. Ajoutez de nouvelles données avec DVC
dvc add data/raw/PlantVillage/NewClass

# 2. Commit et push
git add data/raw/PlantVillage/NewClass.dvc dvc.lock
git commit -m "Add new plant disease data"
git push

# 3. Le workflow se déclenche automatiquement! 🎉
```

### Ce qui se passe automatiquement

1. ✅ **Détection** des changements DVC
2. ✅ **Pull** des données depuis DVC remote (si configuré)
3. ✅ **Exécution** du pipeline DVC (`dvc repro`)
4. ✅ **Entraînement** du modèle
5. ✅ **Enregistrement** dans MLflow
6. ✅ **Construction** de l'image Docker
7. ✅ **Push** vers Docker Hub

## 📊 Vérifier les résultats

### Dans GitHub Actions

1. Allez dans **Actions**
2. Cliquez sur le dernier workflow run
3. Vérifiez chaque job:
   - ✅ Check DVC Changes
   - ✅ Pull Data
   - ✅ Run DVC Pipeline
   - ✅ Build Docker Image
   - ✅ Summary

### Image Docker

L'image est disponible sur Docker Hub:
```
YOUR_USERNAME/plant-disease-mlops:v20241208-123456-abc1234
YOUR_USERNAME/plant-disease-mlops:latest
```

## 🐳 Déployer localement

Une fois le workflow terminé:

```bash
# 1. Pull la nouvelle image
docker pull YOUR_USERNAME/plant-disease-mlops:latest

# 2. Mettre à jour docker-compose.yml
# Changez la ligne image dans docker/docker-compose.yml:
# image: YOUR_USERNAME/plant-disease-mlops:latest

# 3. Redémarrer
docker-compose -f docker/docker-compose.yml up -d
```

## 🔧 Configuration avancée (optionnel)

### MLflow distant

Si vous avez un serveur MLflow:

1. Ajoutez le secret: `MLFLOW_TRACKING_URI = http://votre-serveur:5000`
2. Le workflow utilisera ce serveur au lieu du local

### DVC Remote

Si vos données sont sur S3/Google Drive:

1. Configurez DVC localement:
   ```bash
   dvc remote add -d storage s3://bucket/path
   git add .dvc/config
   git commit -m "Configure DVC remote"
   ```

2. Ajoutez les secrets:
   - `DVC_REMOTE_URL`
   - `DVC_ACCESS_KEY_ID`
   - `DVC_SECRET_ACCESS_KEY`

## 📚 Documentation complète

Pour plus de détails, consultez:
- **`.github/GITHUB_ACTIONS_SETUP.md`** - Guide complet de configuration

## ✅ Checklist

- [ ] Secrets GitHub configurés (`DOCKER_USERNAME`, `DOCKER_PASSWORD`)
- [ ] Workflow testé avec "Run workflow" manuel
- [ ] Image Docker créée et disponible sur Docker Hub
- [ ] Déploiement local testé

## 🎉 C'est prêt!

Votre pipeline est maintenant complètement automatisé via GitHub Actions. Chaque fois que vous ajoutez de nouvelles données avec DVC et poussez vers GitHub, le workflow:

1. Détecte les changements
2. Entraîne le modèle
3. Enregistre dans MLflow
4. Construit et push l'image Docker
5. Prêt pour déploiement!

---

**💡 Astuce**: Utilisez "Run workflow" manuellement pour tester sans ajouter de nouvelles données.

