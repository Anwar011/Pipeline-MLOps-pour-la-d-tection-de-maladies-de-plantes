#!/bin/bash

# ============================================
# Script de Démonstration MLOps - Bash Version
# Pipeline de Détection de Maladies de Plantes
# ============================================

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Fonction pause
pause_demo() {
    echo -e "\n${YELLOW}Appuyez sur Entrée pour continuer...${NC}"
    read -r
}

# Fonction pour afficher un titre
show_title() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# === DÉBUT DE LA DÉMO ===
show_title "🌱 Pipeline MLOps - Détection de Maladies de Plantes 🌱"

echo -e "${WHITE}"
cat << 'EOF'

plant-disease-mlops/
├── src/                 # Code source (API, Training, Models)
├── scripts/             # Scripts MLOps (DVC, Evaluation, Export)
├── docker/              # Dockerfiles
├── k8s/                 # Manifests Kubernetes
├── monitoring/          # Prometheus & Grafana
├── tests/               # Tests unitaires
├── dvc.yaml             # Pipeline DVC
└── config.yaml          # Configuration centralisée

EOF
echo -e "${NC}"

pause_demo

# === 1. STRUCTURE DU PROJET ===
show_title "📁 1. STRUCTURE DU PROJET"

echo -e "${WHITE}Affichage de la structure du projet :${NC}"
ls -la --color=auto
echo ""

pause_demo

# === 2. CONFIGURATION CENTRALISÉE ===
show_title "⚙️  2. CONFIGURATION CENTRALISÉE (config.yaml)"

echo -e "${YELLOW}Contenu du fichier config.yaml :${NC}"
echo ""
head -35 config.yaml
echo -e "${GRAY}... (suite du fichier)${NC}"
echo ""

pause_demo

# === 3. PIPELINE DVC ===
show_title "📊 3. PIPELINE DVC (DataOps)"

echo -e "${GREEN}"
cat << 'EOF'

Pipeline DVC en 5 étapes:
┌─────────────────┐
│ 1. prepare_data │ → Division train/val/test (70/20/10)
└────────┬────────┘
         ▼
┌─────────────────┐
│    2. train     │ → Entraînement CNN avec MLflow
└────────┬────────┘
         ▼
┌─────────────────┐
│   3. evaluate   │ → Métriques (F1, ROC, Confusion Matrix)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. export_model │ → ONNX + Model Registry
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. drift_analysis│ → Détection de dérive (Evidently)
└─────────────────┘

EOF
echo -e "${NC}"

echo -e "${YELLOW}Contenu de dvc.yaml :${NC}"
echo ""
head -30 dvc.yaml
echo ""

pause_demo

# === 4. MODÈLE DEEP LEARNING ===
show_title "🤖 4. MODÈLE DEEP LEARNING (PyTorch Lightning)"

echo -e "${WHITE}"
cat << 'EOF'

Architectures supportées:
┌────────────────────────────────────────────────────────┐
│ CNN                    │ Vision Transformer            │
├────────────────────────┼──────────────────────────────┤
│ • ResNet50             │ • ViT-Base (patch 16x16)     │
│ • EfficientNet-B0      │                              │
│ • VGG16                │                              │
└────────────────────────┴──────────────────────────────┘

Fichier: src/models.py
- PlantDiseaseCNN (LightningModule)
- Métriques: Accuracy, Loss
- Optimizers: Adam, SGD
- Schedulers: Cosine, Step

EOF
echo -e "${NC}"

pause_demo

# === 5. API FASTAPI ===
show_title "🚀 5. API FASTAPI (DeploymentOps)"

echo -e "${WHITE}"
cat << 'EOF'

Endpoints disponibles:
┌──────────────────────────────────────────────────────────┐
│ GET  /              │ Informations sur l'API            │
│ GET  /health        │ Health check (< 2s requis)        │
│ POST /predict       │ Prédiction sur une image          │
│ POST /predict_batch │ Prédiction batch (max 16 images)  │
│ GET  /classes       │ Liste des 15 classes              │
│ GET  /metrics       │ Métriques Prometheus              │
│ GET  /model/info    │ Informations sur le modèle        │
└──────────────────────────────────────────────────────────┘

Fichier: src/api.py
- Prometheus Instrumentator intégré
- CORS middleware
- Validation des fichiers images

EOF
echo -e "${NC}"

pause_demo

# === 6. KUBERNETES ===
show_title "☸️  6. KUBERNETES (Orchestration)"

echo -e "${WHITE}"
cat << 'EOF'

Manifests Kubernetes:
┌─────────────────────────────────────────────────────────┐
│ deployment.yaml  │ 3 replicas, health checks           │
│ service.yaml     │ LoadBalancer, ports 80 & 9090       │
│ hpa.yaml         │ Auto-scaling 2-10 pods (CPU 70%)    │
│ storage.yaml     │ PVC pour modèles et données         │
└─────────────────────────────────────────────────────────┘

EOF
echo -e "${NC}"

echo -e "${YELLOW}Extrait de k8s/deployment.yaml :${NC}"
echo ""
head -25 k8s/deployment.yaml
echo ""

pause_demo

# === 7. CI/CD ===
show_title "🔄 7. CI/CD (GitHub Actions)"

echo -e "${WHITE}"
cat << 'EOF'

Pipeline CI/CD:
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  push/PR ──▶ [Tests] ──▶ [Build Docker] ──▶ [Deploy K8s]     │
│                │              │                 │             │
│                ▼              ▼                 ▼             │
│           pytest         docker push      kubectl apply      │
│           flake8         trivy scan       health check       │
│           black                                               │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Fichier: .github/workflows/mlops-pipeline.yml

EOF
echo -e "${NC}"

pause_demo

# === 8. MONITORING ===
show_title "📈 8. MONITORING (Prometheus + Grafana)"

echo -e "${WHITE}"
cat << 'EOF'

Métriques collectées:
┌─────────────────────────────────────────────────────────────┐
│ Performance API                                              │
│   • api_requests_total (counter)                            │
│   • api_request_duration_seconds (histogram)                │
│   • api_active_requests (gauge)                             │
├─────────────────────────────────────────────────────────────┤
│ Prédictions                                                  │
│   • prediction_confidence (histogram)                       │
│   • predictions_by_class_total (counter)                    │
│   • inference_latency_seconds (histogram)                   │
├─────────────────────────────────────────────────────────────┤
│ Drift Detection (Evidently)                                  │
│   • Data drift report HTML                                  │
│   • Feature drift scores                                    │
└─────────────────────────────────────────────────────────────┘

URLs (après docker-compose up):
  • Grafana:    http://localhost:3000 (admin/admin)
  • Prometheus: http://localhost:9091
  • MLflow:     http://localhost:5000

EOF
echo -e "${NC}"

pause_demo

# === 9. CLASSES SUPPORTÉES ===
show_title "🌿 9. CLASSES DE MALADIES SUPPORTÉES"

echo -e "${GREEN}"
cat << 'EOF'

Dataset PlantVillage - 15 classes:
┌─────────────────────────────────────────────────────────────┐
│ Poivron                                                      │
│   • Pepper__bell___Bacterial_spot                           │
│   • Pepper__bell___healthy                                  │
├─────────────────────────────────────────────────────────────┤
│ Pomme de terre                                               │
│   • Potato___Early_blight                                   │
│   • Potato___Late_blight                                    │
│   • Potato___healthy                                        │
├─────────────────────────────────────────────────────────────┤
│ Tomate                                                       │
│   • Tomato_Bacterial_spot                                   │
│   • Tomato_Early_blight                                     │
│   • Tomato_Late_blight                                      │
│   • Tomato_Leaf_Mold                                        │
│   • Tomato_Septoria_leaf_spot                               │
│   • Tomato_Spider_mites_Two_spotted_spider_mite             │
│   • Tomato__Target_Spot                                     │
│   • Tomato__Tomato_YellowLeaf__Curl_Virus                   │
│   • Tomato__Tomato_mosaic_virus                             │
│   • Tomato_healthy                                          │
└─────────────────────────────────────────────────────────────┘

EOF
echo -e "${NC}"

pause_demo

# === 10. RÉSUMÉ ===
show_title "✅ 10. RÉSUMÉ - CONFORMITÉ AU CAHIER DES CHARGES"

echo -e "${GREEN}"
cat << 'EOF'

┌─────────────────────────────────────────────────────────────┐
│ Exigence                          │ Status │ Fichier       │
├─────────────────────────────────────────────────────────────┤
│ DVC - Gestion données             │   ✅   │ dvc.yaml      │
│ MLflow - Tracking                 │   ✅   │ src/train.py  │
│ PyTorch Lightning                 │   ✅   │ src/models.py │
│ FastAPI - API REST                │   ✅   │ src/api.py    │
│ Docker - Conteneurisation         │   ✅   │ docker/       │
│ Kubernetes - Orchestration        │   ✅   │ k8s/          │
│ GitHub Actions - CI/CD            │   ✅   │ .github/      │
│ Prometheus - Métriques            │   ✅   │ monitoring/   │
│ Grafana - Dashboard               │   ✅   │ grafana/      │
│ Evidently - Drift Detection       │   ✅   │ scripts/      │
│ Tests unitaires                   │   ✅   │ tests/        │
│ Export ONNX                       │   ✅   │ scripts/      │
│ Temps réponse < 2s                │   ✅   │ src/api.py    │
└─────────────────────────────────────────────────────────────┘

EOF
echo -e "${NC}"

# === FIN ===
show_title "🎉 FIN DE LA DÉMONSTRATION 🎉"

echo -e "${CYAN}Le pipeline MLOps est complet et conforme au cahier des charges !${NC}"
echo ""
echo -e "${YELLOW}Pour lancer les commandes réelles, consultez les scripts dans le dossier scripts/${NC}"
echo -e "${YELLOW}- run_pipeline.sh : Pipeline complet${NC}"
echo -e "${YELLOW}- run_api.sh : Lancer l'API${NC}"
echo -e "${YELLOW}- run_monitoring.sh : Stack de monitoring${NC}"
echo ""

echo -e "${GREEN}Bonne soutenance ! 🎓${NC}"