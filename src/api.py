"""
API d'inférence pour la détection de maladies de plantes utilisant FastAPI.
Conforme au cahier des charges: API REST avec monitoring Prometheus.
"""

import base64
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import prometheus_client
import torch
import torch.nn as nn
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, Info
from torchvision import transforms

# Import prometheus-fastapi-instrumentator pour monitoring automatique
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    INSTRUMENTATOR_AVAILABLE = True
except ImportError:
    INSTRUMENTATOR_AVAILABLE = False

from models import create_model

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseInferenceAPI:
    """API d'inférence pour la détection de maladies de plantes."""

    def __init__(self, config_path="config.yaml"):
        """Initialiser l'API avec la configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_config = self.config["api"]
        self.model_config = self.config["model"]

        # Charger le mapping des classes
        self.class_mapping = self.load_class_mapping()

        # Charger le modèle
        self.model = self.load_model()

        # Transformations pour l'inférence
        self.transform = self.get_inference_transform()

        # Métriques Prometheus
        self.setup_metrics()

        logger.info("API d'inférence initialisée")

    def load_class_mapping(self):
        """Charger le mapping des classes."""
        mapping_path = "data/class_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Mapping des classes non trouvé: {mapping_path}")
            return {}

    def load_model(self):
        """Charger le modèle entraîné."""
        model_path = self.api_config["model_path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

        # Charger le modèle selon le type
        model_type = self.model_config.get("architecture", "resnet50")
        if "vit" in model_type.lower():
            model = create_model("vit")
        else:
            model = create_model("cnn")

        # Charger les poids
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Déplacer sur GPU si disponible
        if torch.cuda.is_available():
            model = model.cuda()

        logger.info(f"Modèle chargé depuis {model_path}")
        return model

    def get_inference_transform(self):
        """Définir les transformations pour l'inférence."""
        return transforms.Compose(
            [
                transforms.Resize(self.config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup_metrics(self):
        """Configurer les métriques Prometheus."""
        self.requests_total = Counter(
            "api_requests_total", "Total number of API requests", ["method", "endpoint"]
        )

        self.requests_duration = Histogram(
            "api_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )

        self.prediction_confidence = Histogram(
            "prediction_confidence",
            "Prediction confidence distribution",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.active_requests = Gauge("api_active_requests", "Number of active requests")
        
        # Métriques additionnelles pour le cahier des charges
        self.model_info = Info("model_info", "Information about the loaded model")
        self.model_info.info({
            "architecture": self.model_config.get("architecture", "unknown"),
            "num_classes": str(len(self.class_mapping)),
            "version": "1.0.0"
        })
        
        self.inference_latency = Histogram(
            "inference_latency_seconds",
            "Model inference latency in seconds",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        )
        
        self.predictions_by_class = Counter(
            "predictions_by_class_total",
            "Total predictions per class",
            ["class_name"]
        )

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Prétraite une image pour l'inférence."""
        try:
            # Ouvrir l'image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Appliquer les transformations
            tensor = self.transform(image)

            # Ajouter la dimension batch
            tensor = tensor.unsqueeze(0)

            return tensor
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Erreur de prétraitement: {str(e)}"
            )

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Effectuer une prédiction sur une image."""
        start_time = time.time()

        try:
            self.active_requests.inc()

            # Déplacer sur le bon device
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                # Convertir en valeurs Python
                confidence = confidence.item()
                predicted_class = predicted_class.item()

                # Obtenir le nom de la classe
                class_name = self.class_mapping.get(
                    str(predicted_class), f"Classe_{predicted_class}"
                )

                # Obtenir les top 5 prédictions
                top5_prob, top5_classes = torch.topk(probabilities, 5)
                top5_predictions = []
                for prob, cls in zip(top5_prob[0], top5_classes[0]):
                    cls_name = self.class_mapping.get(
                        str(cls.item()), f"Classe_{cls.item()}"
                    )
                    top5_predictions.append(
                        {"class": cls_name, "confidence": prob.item()}
                    )

            # Logger les métriques
            self.prediction_confidence.observe(confidence)
            self.inference_latency.observe(inference_time)
            self.predictions_by_class.labels(class_name=class_name).inc()

            inference_time = time.time() - start_time

            return {
                "prediction": class_name,
                "confidence": confidence,
                "inference_time": inference_time,
                "top5_predictions": top5_predictions,
                "model_info": {
                    "architecture": self.model_config.get("architecture", "unknown"),
                    "num_classes": len(self.class_mapping),
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Erreur de prédiction: {str(e)}"
            )
        finally:
            self.active_requests.dec()


# Créer l'application FastAPI
app = FastAPI(
    title="Plant Disease Detection API",
    description="""
    ## API MLOps pour la détection automatique de maladies de plantes
    
    Cette API fait partie d'un pipeline MLOps complet incluant:
    - **Entraînement** avec PyTorch Lightning
    - **Tracking** avec MLflow
    - **Monitoring** avec Prometheus/Grafana
    - **Déploiement** avec Docker/Kubernetes
    
    ### Endpoints principaux:
    - `/predict` - Prédiction sur une image
    - `/predict_batch` - Prédiction sur plusieurs images
    - `/health` - Vérification de santé
    - `/metrics` - Métriques Prometheus
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrumentator Prometheus automatique (cahier des charges)
if INSTRUMENTATOR_AVAILABLE:
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics/auto")

# Variable globale pour l'API (initialisée au démarrage)
inference_api = None


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API."""
    global inference_api
    try:
        inference_api = PlantDiseaseInferenceAPI()
        logger.info("✅ API initialisée avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation: {e}")
        # Créer une API en mode dégradé
        inference_api = None


@app.get("/")
async def root():
    """Endpoint racine avec informations sur l'API."""
    if inference_api:
        inference_api.requests_total.labels(method="GET", endpoint="/").inc()
    return {
        "message": "🌱 API de détection de maladies de plantes",
        "version": "1.0.0",
        "status": "ready" if inference_api else "degraded",
        "endpoints": {
            "predict": "/predict (POST) - Prédiction sur une image",
            "predict_batch": "/predict_batch (POST) - Prédiction sur plusieurs images",
            "health": "/health (GET) - Vérification de santé",
            "metrics": "/metrics (GET) - Métriques Prometheus",
            "classes": "/classes (GET) - Liste des classes",
            "docs": "/docs (GET) - Documentation Swagger"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API (cahier des charges: temps < 2s)."""
    if inference_api:
        inference_api.requests_total.labels(method="GET", endpoint="/health").inc()
    
    health_status = {
        "status": "healthy" if inference_api else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "model_loaded": inference_api is not None and inference_api.model is not None,
            "class_mapping_loaded": inference_api is not None and len(inference_api.class_mapping) > 0,
        }
    }
    
    if inference_api:
        health_status["model_info"] = {
            "architecture": inference_api.model_config.get("architecture", "unknown"),
            "num_classes": len(inference_api.class_mapping)
        }
    
    return health_status


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Prédire la maladie d'une plante à partir d'une image.
    
    Conforme au cahier des charges:
    - Temps de réponse < 2s
    - Retourne la prédiction avec confiance
    """
    if inference_api is None:
        raise HTTPException(status_code=503, detail="Service non disponible - modèle non chargé")
    
    with inference_api.requests_duration.labels(
        method="POST", endpoint="/predict"
    ).time():
        inference_api.requests_total.labels(method="POST", endpoint="/predict").inc()

        # Vérifier le type de fichier
        if not file.filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            raise HTTPException(status_code=400, detail="Type de fichier non supporté")

        # Lire le contenu du fichier
        contents = await file.read()

        # Prétraiter l'image
        image_tensor = inference_api.preprocess_image(contents)

        # Effectuer la prédiction
        result = inference_api.predict(image_tensor)

        return result


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Prédire les maladies pour un lot d'images."""
    if inference_api is None:
        raise HTTPException(status_code=503, detail="Service non disponible")
    
    with inference_api.requests_duration.labels(
        method="POST", endpoint="/predict_batch"
    ).time():
        inference_api.requests_total.labels(
            method="POST", endpoint="/predict_batch"
        ).inc()

        if len(files) > inference_api.api_config["max_batch_size"]:
            raise HTTPException(
                status_code=400,
                detail=f"Nombre maximum d'images dépassé: {inference_api.api_config['max_batch_size']}",
            )

        results = []
        for file in files:
            try:
                contents = await file.read()
                image_tensor = inference_api.preprocess_image(contents)
                result = inference_api.predict(image_tensor)
                result["filename"] = file.filename
                results.append(result)
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

        return {"results": results, "total": len(results)}


from fastapi import Response


@app.get("/metrics")
async def metrics():
    """Exporter les métriques Prometheus (cahier des charges: monitoring)."""
    return Response(
        content=prometheus_client.generate_latest(), media_type="text/plain"
    )


@app.get("/classes")
async def get_classes():
    """Obtenir la liste des classes supportées."""
    if inference_api is None:
        raise HTTPException(status_code=503, detail="Service non disponible")
    
    inference_api.requests_total.labels(method="GET", endpoint="/classes").inc()
    return {
        "classes": list(inference_api.class_mapping.values()),
        "num_classes": len(inference_api.class_mapping),
        "class_mapping": inference_api.class_mapping
    }


@app.get("/model/info")
async def get_model_info():
    """Obtenir les informations sur le modèle chargé."""
    if inference_api is None:
        raise HTTPException(status_code=503, detail="Service non disponible")
    
    return {
        "architecture": inference_api.model_config.get("architecture", "unknown"),
        "num_classes": len(inference_api.class_mapping),
        "image_size": inference_api.config["data"]["image_size"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": inference_api.api_config.get("model_path", "unknown")
    }


def main():
    """Fonction principale pour démarrer l'API."""
    host = inference_api.api_config["host"]
    port = inference_api.api_config["port"]

    logger.info(f"Démarrage de l'API sur {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
# Workflow test - 08 دجنبر, 2025 +01 19:35:55
# Test deployment trigger - 08 دجنبر, 2025 +01 23:47:58
