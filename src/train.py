"""
Script principal d'entraînement des modèles avec suivi MLflow.
"""

import os
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import mlflow.pytorch
from pathlib import Path
import logging
from datetime import datetime

from data_preprocessing import DataPreprocessor
from models import create_model, get_training_callbacks

logger = logging.getLogger(__name__)

def setup_mlflow(config):
    """Configurer MLflow pour le suivi des expériences."""
    mlflow_config = config['mlflow']

    # Définir l'URI de suivi
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])

    # Créer/définir l'expérience
    experiment_name = mlflow_config['experiment_name']
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    # Créer le logger MLflow pour PyTorch Lightning
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_config['tracking_uri']
    )

    logger.info(f"MLflow configuré - Expérience: {experiment_name}")
    return mlf_logger

def log_parameters(config, model_type):
    """Logger les paramètres dans MLflow."""
    # Paramètres de données
    data_config = config['data']
    mlflow.log_param("batch_size", data_config['batch_size'])
    mlflow.log_param("image_size", data_config['image_size'])
    mlflow.log_param("train_split", data_config['train_split'])
    mlflow.log_param("val_split", data_config['val_split'])
    mlflow.log_param("test_split", data_config['test_split'])

    # Paramètres du modèle
    model_config = config['model']
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("architecture", model_config['architecture'])
    mlflow.log_param("num_classes", model_config['num_classes'])
    mlflow.log_param("pretrained", model_config['pretrained'])
    mlflow.log_param("freeze_backbone", model_config['freeze_backbone'])

    # Paramètres d'entraînement
    training_config = config['training']
    mlflow.log_param("epochs", training_config['epochs'])
    mlflow.log_param("learning_rate", training_config['learning_rate'])
    mlflow.log_param("weight_decay", training_config['weight_decay'])
    mlflow.log_param("optimizer", training_config['optimizer'])
    mlflow.log_param("scheduler", training_config['scheduler'])

def train_model(model_type, dataset_path, config_path="config.yaml"):
    """
    Fonction principale d'entraînement.
    Args:
        model_type (str): Type de modèle ('cnn' ou 'vit')
        dataset_path (str): Chemin vers le dataset
        config_path (str): Chemin vers la configuration
    """
    # Charger la configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Début de l'entraînement du modèle {model_type}")

    # Préparer les données
    logger.info("Préparation des données...")
    preprocessor = DataPreprocessor(config_path)
    data_result = preprocessor.create_data_loaders(dataset_path)

    data_loaders = data_result['data_loaders']
    num_classes = data_result['num_classes']
    class_names = data_result['class_names']

    # Mettre à jour le nombre de classes dans la config
    config['model']['num_classes'] = num_classes

    # Créer le modèle
    logger.info(f"Création du modèle {model_type}...")
    model = create_model(model_type, config_path)

    # Configurer MLflow
    mlf_logger = setup_mlflow(config)

    # Démarrer un run MLflow
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Logger les paramètres
        log_parameters(config, model_type)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", str(class_names))

        # Callbacks
        callbacks = get_training_callbacks(config_path)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=config['training']['epochs'],
            accelerator="auto",
            devices="auto",
            logger=mlf_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Entraînement
        logger.info("Début de l'entraînement...")
        trainer.fit(model, data_loaders['train'], data_loaders['val'])

        # Évaluation sur le test set
        logger.info("Évaluation sur le jeu de test...")
        trainer.test(model, data_loaders['test'])

        # Sauvegarder le modèle dans MLflow
        logger.info("Sauvegarde du modèle dans MLflow...")
        mlflow.pytorch.log_model(model, "model")

        # Logger des métriques finales
        final_metrics = trainer.callback_metrics
        for metric_name, metric_value in final_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                mlflow.log_metric(f"final_{metric_name}", metric_value.item())
            else:
                mlflow.log_metric(f"final_{metric_name}", metric_value)

        logger.info("Entraînement terminé avec succès!")

        return model, trainer

def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles de détection de maladies de plantes")
    parser.add_argument("--model", type=str, choices=['cnn', 'vit'], default='cnn',
                       help="Type de modèle à entraîner")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Chemin vers le dataset")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--gpu", type=int, default=None,
                       help="ID du GPU à utiliser")

    args = parser.parse_args()

    # Configuration du GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        train_model(args.model, args.dataset, args.config)
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise

if __name__ == "__main__":
    main()
