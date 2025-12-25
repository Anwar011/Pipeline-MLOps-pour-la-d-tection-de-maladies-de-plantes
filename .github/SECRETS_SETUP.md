# GitHub Actions Secrets Configuration

Ce document décrit les secrets GitHub nécessaires pour le pipeline MLOps.

## 🔐 Secrets Requis

### Docker Hub
| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Votre nom d'utilisateur Docker Hub |
| `DOCKER_PASSWORD` | Votre token d'accès Docker Hub (pas le mot de passe) |

### MLflow (Optionnel - pour serveur distant)
| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow (ex: `https://mlflow.example.com`) |
| `MLFLOW_TRACKING_USERNAME` | Nom d'utilisateur MLflow (si authentification) |
| `MLFLOW_TRACKING_PASSWORD` | Mot de passe MLflow (si authentification) |

### DVC Remote (Optionnel - selon votre remote)

#### Pour AWS S3
| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | Clé d'accès AWS |
| `AWS_SECRET_ACCESS_KEY` | Clé secrète AWS |

#### Pour Google Drive
| Secret | Description |
|--------|-------------|
| `GDRIVE_CREDENTIALS` | Contenu JSON du fichier de credentials service account |

### Kubernetes (Pour le déploiement)
| Secret | Description |
|--------|-------------|
| `KUBE_CONFIG` | Contenu base64 de votre fichier kubeconfig |

## 📝 Comment Configurer

### 1. Docker Hub Token
1. Allez sur [Docker Hub](https://hub.docker.com/)
2. Settings → Security → New Access Token
3. Copiez le token généré
4. Dans GitHub: Settings → Secrets → New repository secret

### 2. Kubeconfig (pour Kubernetes)
```bash
# Encoder votre kubeconfig en base64
cat ~/.kube/config | base64 -w 0
# Copiez le résultat dans le secret KUBE_CONFIG
```

### 3. Configurer les secrets dans GitHub
1. Allez dans votre repository GitHub
2. Settings → Secrets and variables → Actions
3. Cliquez "New repository secret"
4. Ajoutez chaque secret

## 🔄 Workflow Déclenché

Le pipeline `mlops-data-driven.yml` se déclenche automatiquement quand:

1. **`dvc.lock` change** → Nouvelles données détectées par DVC
2. **`dvc.yaml` change** → Pipeline DVC modifié
3. **`src/train.py` ou `src/models.py` change** → Code d'entraînement modifié
4. **Déclenchement manuel** → Via l'interface GitHub Actions

## 📊 Flux du Pipeline

```
┌─────────────────┐
│  DVC detecte    │
│  nouvelles      │
│  données        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pull données   │
│  depuis DVC     │
│  remote         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Entraînement   │
│  du modèle      │
│  → MLflow       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build image    │
│  API avec       │
│  nouveau modèle │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Push vers      │
│  Docker Hub     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deploy vers    │
│  Kubernetes     │
│  (si main)      │
└─────────────────┘
```

## 🧪 Test Local

Pour tester le pipeline localement:

```bash
# 1. Simuler un changement DVC
echo "test" >> data/test.txt
dvc add data/test.txt
git add data/test.txt.dvc dvc.lock

# 2. Commit et push
git commit -m "Add new data"
git push

# Le pipeline se déclenche automatiquement!
```

## 🔍 Debugging

Si le pipeline échoue:

1. **Check-changes échoue**: Vérifiez que les fichiers DVC sont bien commités
2. **Pull-data échoue**: Vérifiez les credentials DVC remote
3. **Train échoue**: Vérifiez les dépendances et le format des données
4. **Build-api échoue**: Vérifiez que le modèle est bien sauvegardé
5. **Deploy échoue**: Vérifiez le kubeconfig et les permissions
