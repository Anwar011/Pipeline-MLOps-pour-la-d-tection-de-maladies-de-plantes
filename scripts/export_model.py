#!/usr/bin/env python3
"""
Script d'export du modèle en ONNX et TorchScript.
Conforme au cahier des charges: Registry et packaging (ONNX/TorchScript).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import torch
import yaml

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model(config_path: str = "config.yaml"):
    """
    Exporte le modèle en différents formats pour la production.
    
    Formats d'export:
    - PyTorch checkpoint (.ckpt)
    - ONNX (.onnx) - pour l'inférence optimisée
    - TorchScript (.pt) - pour le déploiement
    """
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Créer le dossier de production
    production_dir = Path("models/production")
    production_dir.mkdir(parents=True, exist_ok=True)
    
    # Trouver le meilleur checkpoint
    checkpoint_dir = Path(config["training"]["checkpoint_path"])
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    if not checkpoints:
        logger.error("❌ Aucun checkpoint trouvé pour l'export!")
        logger.info("💡 Entraînez d'abord un modèle avec: python src/train.py")
        return None
    
    # Prendre le checkpoint le plus récent (ou le meilleur basé sur le nom)
    best_checkpoint = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"📂 Chargement du checkpoint: {best_checkpoint}")
    
    # Charger le modèle
    model = create_model("cnn", config_path)
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # 1. Export PyTorch checkpoint pour production
    production_ckpt = production_dir / "model.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "architecture": config["model"]["architecture"],
        "num_classes": config["model"]["num_classes"]
    }, production_ckpt)
    logger.info(f"✅ Checkpoint exporté: {production_ckpt}")
    
    # 2. Export ONNX
    export_onnx(model, config, production_dir)
    
    # 3. Export TorchScript
    export_torchscript(model, config, production_dir)
    
    # 4. Enregistrer dans MLflow Model Registry
    register_model_mlflow(model, config)
    
    logger.info("✅ Export du modèle terminé!")
    
    return {
        "checkpoint": str(production_ckpt),
        "onnx": str(production_dir / "model.onnx"),
        "torchscript": str(production_dir / "model.pt")
    }


def export_onnx(model, config, output_dir):
    """Exporte le modèle au format ONNX."""
    try:
        onnx_path = output_dir / "model.onnx"
        
        # Créer un input dummy
        image_size = config["data"]["image_size"]
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
        
        # Export ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        
        logger.info(f"✅ Modèle ONNX exporté: {onnx_path}")
        
        # Vérifier l'export ONNX
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✅ Modèle ONNX validé")
        except ImportError:
            logger.warning("⚠️  onnx non installé, validation ignorée")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'export ONNX: {e}")


def export_torchscript(model, config, output_dir):
    """Exporte le modèle au format TorchScript."""
    try:
        script_path = output_dir / "model.pt"
        
        # Créer un input dummy
        image_size = config["data"]["image_size"]
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
        
        # Export TorchScript via tracing
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(script_path))
        
        logger.info(f"✅ Modèle TorchScript exporté: {script_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'export TorchScript: {e}")


def register_model_mlflow(model, config):
    """Enregistre le modèle dans MLflow Model Registry."""
    try:
        mlflow_config = config["mlflow"]
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Créer un run pour l'enregistrement
        with mlflow.start_run(run_name="model_registration"):
            # Logger le modèle
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name="plant_disease_model"
            )
            
            # Logger les paramètres
            mlflow.log_params({
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "image_size": str(config["data"]["image_size"])
            })
        
        logger.info("✅ Modèle enregistré dans MLflow Model Registry")
        
    except Exception as e:
        logger.warning(f"⚠️  Impossible d'enregistrer dans MLflow: {e}")
        logger.info("💡 Assurez-vous que le serveur MLflow est démarré")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporter le modèle")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    export_model(args.config)
