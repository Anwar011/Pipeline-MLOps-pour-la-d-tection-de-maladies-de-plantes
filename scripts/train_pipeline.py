#!/usr/bin/env python3
"""
Script pour exécuter le pipeline complet d'entraînement MLOps.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import DataPreprocessor
from models import create_model
from train import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline(dataset_path, model_type="cnn", config_path="config.yaml"):
    """
    Exécute le pipeline complet d'entraînement.

    Args:
        dataset_path (str): Chemin vers le dataset
        model_type (str): Type de modèle ('cnn' ou 'vit')
        config_path (str): Chemin vers la configuration
    """
    logger.info("🚀 Démarrage du pipeline MLOps de détection de maladies de plantes")

    try:
        # 1. Préparation des données
        logger.info("📊 Étape 1: Préparation des données")
        preprocessor = DataPreprocessor(config_path)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset non trouvé: {dataset_path}")

        # Créer les DataLoaders
        data_result = preprocessor.create_data_loaders(dataset_path)
        logger.info(f"✅ Données préparées: {data_result['num_classes']} classes trouvées")

        # 2. Entraînement du modèle
        logger.info("🤖 Étape 2: Entraînement du modèle")
        model, trainer = train_model(model_type, dataset_path, config_path)
        logger.info("✅ Entraînement terminé")

        # 3. Évaluation
        logger.info("📈 Étape 3: Évaluation du modèle")
        # L'évaluation est déjà faite dans train_model

        logger.info("🎉 Pipeline d'entraînement terminé avec succès!")

        return model, trainer

    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement MLOps")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Chemin vers le dataset PlantVillage"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['cnn', 'vit'],
        default='cnn',
        help="Type de modèle à entraîner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration"
    )

    args = parser.parse_args()

    # Vérifier que le dataset existe
    if not os.path.exists(args.dataset):
        logger.error(f"❌ Dataset non trouvé: {args.dataset}")
        logger.info("💡 Téléchargez le dataset PlantVillage depuis Kaggle:")
        logger.info("   https://www.kaggle.com/datasets/emmarex/plantdisease")
        sys.exit(1)

    try:
        run_training_pipeline(args.dataset, args.model, args.config)
    except Exception as e:
        logger.error(f"❌ Échec du pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
