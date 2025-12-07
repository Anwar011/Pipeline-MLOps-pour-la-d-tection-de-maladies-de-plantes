import pytest
import torch
import yaml
from pathlib import Path
from src.models import create_model
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_config_structure(config):
    """Test if config has required sections."""
    assert "model" in config
    assert "data" in config
    assert "training" in config
    assert "api" in config

def test_model_creation(config):
    """Test if model can be created with config."""
    # Create a dummy config file for the test
    # or just pass the path since create_model expects a path
    model = create_model("cnn", "config.yaml")
    assert model is not None
    assert hasattr(model, "forward")

def test_model_output_shape(config):
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
        
    assert output.shape == (batch_size, num_classes)
