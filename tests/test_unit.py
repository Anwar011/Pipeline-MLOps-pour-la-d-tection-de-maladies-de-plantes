"""
Tests unitaires pour le projet MLOps de détection de maladies de plantes.
Conforme au cahier des charges: Tests unitaires.
"""

import pytest
import torch
import yaml
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import create_model
from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def config():
    """Charger la configuration."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ============================================
# Tests de Configuration
# ============================================

class TestConfiguration:
    """Tests pour la configuration du projet."""

    def test_config_structure(self, config):
        """Test if config has required sections (cahier des charges)."""
        assert "model" in config, "Section 'model' manquante"
        assert "data" in config, "Section 'data' manquante"
        assert "training" in config, "Section 'training' manquante"
        assert "api" in config, "Section 'api' manquante"
        assert "mlflow" in config, "Section 'mlflow' manquante"

    def test_data_splits_sum_to_one(self, config):
        """Test que les splits de données totalisent 1.0."""
        total = config['data']['train_split'] + config['data']['val_split'] + config['data']['test_split']
        assert abs(total - 1.0) < 0.01, f"Les splits totalisent {total}, pas 1.0"

    def test_model_config_valid(self, config):
        """Test que la config du modèle est valide."""
        assert config['model']['num_classes'] > 0
        assert config['model']['architecture'] in ['resnet50', 'efficientnet_b0', 'vgg16', 'vit']


# ============================================
# Tests de Modèle
# ============================================

class TestModel:
    """Tests pour les modèles de deep learning."""

    def test_model_creation(self, config):
        """Test if model can be created with config."""
        model = create_model("cnn", "config.yaml")
        assert model is not None
        assert hasattr(model, "forward")

    def test_model_output_shape(self, config):
        """Test model forward pass output shape."""
        model = create_model("cnn", "config.yaml")
        batch_size = 2
        channels = 3
        height = config['data']['image_size'][0]
        width = config['data']['image_size'][1]
        num_classes = config['model']['num_classes']
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            
        assert output.shape == (batch_size, num_classes), \
            f"Shape incorrect: {output.shape} vs ({batch_size}, {num_classes})"

    def test_model_parameters_trainable(self, config):
        """Test que le modèle a des paramètres entraînables."""
        model = create_model("cnn", "config.yaml")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0, "Le modèle n'a pas de paramètres entraînables"

    def test_model_forward_deterministic(self, config):
        """Test que le forward pass est déterministe en mode eval."""
        model = create_model("cnn", "config.yaml")
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2), "Le modèle n'est pas déterministe en mode eval"


# ============================================
# Tests de Data Preprocessing
# ============================================

class TestDataPreprocessing:
    """Tests pour le prétraitement des données."""

    def test_preprocessor_initialization(self):
        """Test que le preprocessor s'initialise correctement."""
        preprocessor = DataPreprocessor("config.yaml")
        assert preprocessor is not None

    def test_augmentation_transform_training(self):
        """Test que les augmentations d'entraînement fonctionnent."""
        preprocessor = DataPreprocessor("config.yaml")
        transform = preprocessor.get_data_augmentation(is_training=True)
        assert transform is not None

    def test_augmentation_transform_inference(self):
        """Test que les transformations d'inférence fonctionnent."""
        preprocessor = DataPreprocessor("config.yaml")
        transform = preprocessor.get_data_augmentation(is_training=False)
        assert transform is not None


# ============================================
# Tests de Pipeline DVC
# ============================================

class TestDVCPipeline:
    """Tests pour le pipeline DVC."""

    def test_dvc_yaml_exists(self):
        """Test que dvc.yaml existe."""
        assert Path("dvc.yaml").exists(), "dvc.yaml non trouvé"

    def test_dvc_yaml_valid(self):
        """Test que dvc.yaml est valide."""
        with open("dvc.yaml", "r") as f:
            dvc_config = yaml.safe_load(f)
        
        assert "stages" in dvc_config, "Section 'stages' manquante dans dvc.yaml"
        
        # Vérifier les stages requis
        required_stages = ["prepare_data", "train", "evaluate"]
        for stage in required_stages:
            assert stage in dvc_config["stages"], f"Stage '{stage}' manquant"


# ============================================
# Tests d'Intégration
# ============================================

class TestIntegration:
    """Tests d'intégration."""

    def test_full_inference_pipeline(self, config):
        """Test du pipeline d'inférence complet."""
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        
        # Créer une image de test
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Charger le modèle
        model = create_model("cnn", "config.yaml")
        model.eval()
        
        # Transformer l'image
        transform = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        
        # Prédiction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Vérifications
        assert 0 <= confidence.item() <= 1, "Confiance hors limites"
        assert 0 <= predicted.item() < config['model']['num_classes'], "Classe prédite invalide"


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src"])
