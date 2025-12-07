"""
API d'inférence pour la détection de maladies de plantes utilisant FastAPI.
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import prometheus_client
import torch
import torch.nn as nn
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram
from torchvision import transforms

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
    description="API pour la détection automatique de maladies de plantes",
    version="1.0.0",
)

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser l'API d'inférence
inference_api = PlantDiseaseInferenceAPI()


@app.get("/")
async def root():
    """Endpoint racine."""
    inference_api.requests_total.labels(method="GET", endpoint="/").inc()
    return {
        "message": "API de détection de maladies de plantes",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)",
        },
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API."""
    inference_api.requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Prédire la maladie d'une plante à partir d'une image."""
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

        return {"results": results}


from fastapi import Response


@app.get("/metrics")
async def metrics():
    """Exporter les métriques Prometheus."""
    return Response(
        content=prometheus_client.generate_latest(), media_type="text/plain"
    )


@app.get("/classes")
async def get_classes():
    """Obtenir la liste des classes supportées."""
    inference_api.requests_total.labels(method="GET", endpoint="/classes").inc()
    return {
        "classes": list(inference_api.class_mapping.values()),
        "num_classes": len(inference_api.class_mapping),
    }


def main():
    """Fonction principale pour démarrer l'API."""
    host = inference_api.api_config["host"]
    port = inference_api.api_config["port"]

    logger.info(f"Démarrage de l'API sur {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
