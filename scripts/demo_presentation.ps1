# ============================================
# Script de Démonstration MLOps
# À exécuter pendant la présentation
# ============================================

Write-Host @"
╔═══════════════════════════════════════════════════════════════════╗
║     🌱 Pipeline MLOps - Détection de Maladies de Plantes 🌱       ║
║                    DÉMONSTRATION INTERACTIVE                       ║
╚═══════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# === FONCTION PAUSE ===
function Pause-Demo {
    Write-Host "`nAppuyez sur une touche pour continuer..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# === 1. STRUCTURE DU PROJET ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "📁 1. STRUCTURE DU PROJET" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

plant-disease-mlops/
├── src/                 # Code source (API, Training, Models)
├── scripts/             # Scripts MLOps (DVC, Evaluation, Export)
├── docker/              # Dockerfiles
├── k8s/                 # Manifests Kubernetes
├── monitoring/          # Prometheus & Grafana
├── tests/               # Tests unitaires
├── dvc.yaml             # Pipeline DVC
└── config.yaml          # Configuration centralisée

"@ -ForegroundColor White

Pause-Demo

# === 2. CONFIGURATION CENTRALISÉE ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "⚙️  2. CONFIGURATION CENTRALISÉE (config.yaml)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Get-Content config.yaml | Select-Object -First 35
Write-Host "`n... (suite du fichier)" -ForegroundColor Gray

Pause-Demo

# === 3. PIPELINE DVC ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "📊 3. PIPELINE DVC (DataOps)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor Green

Write-Host "Contenu de dvc.yaml:" -ForegroundColor Yellow
Get-Content dvc.yaml | Select-Object -First 30

Pause-Demo

# === 4. MODÈLE DEEP LEARNING ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "🤖 4. MODÈLE DEEP LEARNING (PyTorch Lightning)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor White

Pause-Demo

# === 5. API FASTAPI ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "🚀 5. API FASTAPI (DeploymentOps)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor White

Pause-Demo

# === 6. KUBERNETES ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "☸️  6. KUBERNETES (Orchestration)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

Manifests Kubernetes:
┌─────────────────────────────────────────────────────────┐
│ deployment.yaml  │ 3 replicas, health checks           │
│ service.yaml     │ LoadBalancer, ports 80 & 9090       │
│ hpa.yaml         │ Auto-scaling 2-10 pods (CPU 70%)    │
│ storage.yaml     │ PVC pour modèles et données         │
└─────────────────────────────────────────────────────────┘

"@ -ForegroundColor White

Write-Host "Extrait de k8s/deployment.yaml:" -ForegroundColor Yellow
Get-Content k8s/deployment.yaml | Select-Object -First 25

Pause-Demo

# === 7. CI/CD ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "🔄 7. CI/CD (GitHub Actions)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor White

Pause-Demo

# === 8. MONITORING ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "📈 8. MONITORING (Prometheus + Grafana)" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor White

Pause-Demo

# === 9. CLASSES SUPPORTÉES ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "🌿 9. CLASSES DE MALADIES SUPPORTÉES" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor Green

Pause-Demo

# === 10. RÉSUMÉ ===
Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "✅ 10. RÉSUMÉ - CONFORMITÉ AU CAHIER DES CHARGES" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Magenta

Write-Host @"

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

"@ -ForegroundColor Green

Write-Host @"
╔═══════════════════════════════════════════════════════════════════╗
║                    🎉 FIN DE LA DÉMONSTRATION 🎉                  ║
║                                                                    ║
║  Le pipeline MLOps est complet et conforme au cahier des charges  ║
╚═══════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
