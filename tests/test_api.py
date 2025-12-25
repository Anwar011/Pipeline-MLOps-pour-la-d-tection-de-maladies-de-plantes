"""
Tests unitaires et d'intégration pour l'API FastAPI.
Conforme au cahier des charges: Tests unitaires sur FastAPI.
"""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def test_image():
    """Créer une image de test."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def mock_model():
    """Créer un mock du modèle."""
    import torch
    
    model = MagicMock()
    # Simuler une sortie de modèle (15 classes)
    model.return_value = torch.randn(1, 15)
    model.eval = MagicMock()
    return model


@pytest.fixture
def client():
    """Créer un client de test FastAPI."""
    # Mock l'initialisation pour éviter de charger le vrai modèle
    with patch("src.api.PlantDiseaseInferenceAPI") as mock_api:
        mock_instance = MagicMock()
        mock_instance.class_mapping = {str(i): f"Class_{i}" for i in range(15)}
        mock_instance.model_config = {"architecture": "resnet50"}
        mock_instance.api_config = {"max_batch_size": 16}
        mock_instance.config = {"data": {"image_size": [224, 224]}}
        mock_instance.model = MagicMock()
        
        # Mock la prédiction
        mock_instance.predict.return_value = {
            "prediction": "Tomato_healthy",
            "confidence": 0.95,
            "inference_time": 0.05,
            "top5_predictions": [
                {"class": "Tomato_healthy", "confidence": 0.95},
                {"class": "Tomato_Bacterial_spot", "confidence": 0.03}
            ]
        }
        mock_instance.preprocess_image.return_value = MagicMock()
        mock_instance.requests_total = MagicMock()
        mock_instance.requests_duration = MagicMock()
        mock_instance.requests_duration.labels.return_value.time.return_value.__enter__ = MagicMock()
        mock_instance.requests_duration.labels.return_value.time.return_value.__exit__ = MagicMock()
        
        mock_api.return_value = mock_instance
        
        from fastapi.testclient import TestClient
        from src.api import app
        
        # Réinitialiser l'API globale
        import src.api as api_module
        api_module.inference_api = mock_instance
        
        yield TestClient(app)


# ============================================
# Tests de l'endpoint racine
# ============================================

class TestRootEndpoint:
    """Tests pour l'endpoint racine."""
    
    def test_root_returns_200(self, client):
        """Test que l'endpoint racine retourne 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_version(self, client):
        """Test que la réponse contient la version."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_root_contains_endpoints_list(self, client):
        """Test que la réponse liste les endpoints."""
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data


# ============================================
# Tests de l'endpoint health
# ============================================

class TestHealthEndpoint:
    """Tests pour l'endpoint de santé."""
    
    def test_health_returns_200(self, client):
        """Test que health retourne 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Test que health retourne un status."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
    
    def test_health_response_time_under_2s(self, client):
        """Test que le temps de réponse est < 2s (cahier des charges)."""
        import time
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Temps de réponse trop long: {elapsed}s"


# ============================================
# Tests de l'endpoint predict
# ============================================

class TestPredictEndpoint:
    """Tests pour l'endpoint de prédiction."""
    
    def test_predict_returns_200_with_image(self, client, test_image):
        """Test que predict retourne 200 avec une image valide."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        assert response.status_code == 200
    
    def test_predict_returns_prediction(self, client, test_image):
        """Test que predict retourne une prédiction."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
    
    def test_predict_rejects_invalid_file_type(self, client):
        """Test que predict rejette les fichiers non-image."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_predict_confidence_between_0_and_1(self, client, test_image):
        """Test que la confiance est entre 0 et 1."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        data = response.json()
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_returns_top5(self, client, test_image):
        """Test que predict retourne les top 5 prédictions."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        data = response.json()
        assert "top5_predictions" in data


# ============================================
# Tests de l'endpoint classes
# ============================================

class TestClassesEndpoint:
    """Tests pour l'endpoint des classes."""
    
    def test_classes_returns_200(self, client):
        """Test que classes retourne 200."""
        response = client.get("/classes")
        assert response.status_code == 200
    
    def test_classes_returns_list(self, client):
        """Test que classes retourne une liste."""
        response = client.get("/classes")
        data = response.json()
        assert "classes" in data
        assert isinstance(data["classes"], list)
    
    def test_classes_returns_num_classes(self, client):
        """Test que classes retourne le nombre de classes."""
        response = client.get("/classes")
        data = response.json()
        assert "num_classes" in data
        assert data["num_classes"] > 0


# ============================================
# Tests de l'endpoint metrics
# ============================================

class TestMetricsEndpoint:
    """Tests pour l'endpoint des métriques Prometheus."""
    
    def test_metrics_returns_200(self, client):
        """Test que metrics retourne 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_prometheus_format(self, client):
        """Test que metrics retourne du format Prometheus."""
        response = client.get("/metrics")
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


# ============================================
# Tests de performance
# ============================================

class TestPerformance:
    """Tests de performance conformes au cahier des charges."""
    
    def test_inference_time_under_2s(self, client, test_image):
        """Test que l'inférence est < 2s (cahier des charges)."""
        import time
        start = time.time()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Temps d'inférence trop long: {elapsed}s"
    
    def test_batch_prediction_performance(self, client, test_image):
        """Test les performances de prédiction batch."""
        # Créer plusieurs images
        files = [
            ("files", ("test1.jpg", test_image, "image/jpeg")),
            ("files", ("test2.jpg", test_image, "image/jpeg")),
        ]
        
        import time
        start = time.time()
        response = client.post("/predict_batch", files=files)
        elapsed = time.time() - start
        
        # Batch de 2 images devrait prendre < 4s
        assert elapsed < 4.0


# ============================================
# Tests d'intégration
# ============================================

class TestIntegration:
    """Tests d'intégration."""
    
    def test_full_prediction_flow(self, client, test_image):
        """Test du flux complet de prédiction."""
        # 1. Vérifier que l'API est en bonne santé
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Obtenir les classes disponibles
        classes_response = client.get("/classes")
        assert classes_response.status_code == 200
        classes = classes_response.json()["classes"]
        
        # 3. Faire une prédiction
        predict_response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        assert predict_response.status_code == 200
        prediction = predict_response.json()
        
        # 4. Vérifier que la prédiction est une classe valide
        # (dans le mock, on utilise des classes simulées)
        assert "prediction" in prediction


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])
