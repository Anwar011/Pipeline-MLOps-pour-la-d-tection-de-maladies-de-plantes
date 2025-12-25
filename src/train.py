"""
Script principal d'entraînement des modèles avec suivi MLflow.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import MLFlowLogger

from data_preprocessing import DataPreprocessor
from models import create_model, get_training_callbacks

logger = logging.getLogger(__name__)


def setup_mlflow(config):
    """Configurer MLflow pour le suivi des expériences."""
    mlflow_config = config["mlflow"]

    # Définir l'URI de suivi
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    # Créer/définir l'expérience
    experiment_name = mlflow_config["experiment_name"]
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    # Créer le logger MLflow pour PyTorch Lightning
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=mlflow_config["tracking_uri"]
    )

    logger.info(f"MLflow configuré - Expérience: {experiment_name}")
    return mlf_logger


def log_parameters(config, model_type):
    """Logger les paramètres dans MLflow."""
    # Paramètres de données
    data_config = config["data"]
    mlflow.log_param("batch_size", data_config["batch_size"])
    mlflow.log_param("image_size", data_config["image_size"])
    mlflow.log_param("train_split", data_config["train_split"])
    mlflow.log_param("val_split", data_config["val_split"])
    mlflow.log_param("test_split", data_config["test_split"])

    # Paramètres du modèle
    model_config = config["model"]
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("architecture", model_config["architecture"])
    mlflow.log_param("num_classes", model_config["num_classes"])
    mlflow.log_param("pretrained", model_config["pretrained"])
    mlflow.log_param("freeze_backbone", model_config["freeze_backbone"])

    # Paramètres d'entraînement
    training_config = config["training"]
    mlflow.log_param("epochs", training_config["epochs"])
    mlflow.log_param("learning_rate", training_config["learning_rate"])
    mlflow.log_param("weight_decay", training_config["weight_decay"])
    mlflow.log_param("optimizer", training_config["optimizer"])
    mlflow.log_param("scheduler", training_config["scheduler"])


def train_model(model_type, dataset_path, config_path="config.yaml"):
    """
    Fonction principale d'entraînement.
    Args:
        model_type (str): Type de modèle ('cnn' ou 'vit')
        dataset_path (str): Chemin vers le dataset
        config_path (str): Chemin vers la configuration
    """
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Début de l'entraînement du modèle {model_type}")

    # Préparer les données
    logger.info("Préparation des données...")
    preprocessor = DataPreprocessor(config_path)
    data_result = preprocessor.create_data_loaders(dataset_path)

    data_loaders = data_result["data_loaders"]
    num_classes = data_result["num_classes"]
    class_names = data_result["class_names"]

    # Mettre à jour le nombre de classes dans la config
    config["model"]["num_classes"] = num_classes

    # Créer le modèle
    logger.info(f"Création du modèle {model_type}...")
    model = create_model(model_type, config_path)

    # Configurer MLflow
    mlf_logger = setup_mlflow(config)

    # Démarrer un run MLflow
    with mlflow.start_run(
        run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        # Logger les paramètres
        log_parameters(config, model_type)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", str(class_names))
        
        # Logger les informations sur les données (cahier des charges: enregistrement des données)
        try:
            # Compter les images par classe
            from pathlib import Path
            dataset_path_obj = Path(dataset_path)
            if dataset_path_obj.exists():
                class_counts = {}
                for class_dir in dataset_path_obj.iterdir():
                    if class_dir.is_dir():
                        image_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                        class_counts[class_dir.name] = image_count
                
                mlflow.log_dict(class_counts, "data/class_counts.json")
                total_images = sum(class_counts.values())
                mlflow.log_param("total_images", total_images)
                mlflow.log_param("data_version", str(dataset_path_obj.stat().st_mtime))
                logger.info(f"📊 Données enregistrées dans MLflow: {total_images} images, {len(class_counts)} classes")
        except Exception as e:
            logger.warning(f"⚠️  Impossible d'enregistrer les détails des données: {e}")

        # Callbacks
        callbacks = get_training_callbacks(config_path)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="auto",
            devices="auto",
            logger=mlf_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Entraînement
        logger.info("Début de l'entraînement...")
        trainer.fit(model, data_loaders["train"], data_loaders["val"])

        # Évaluation sur le test set
        logger.info("Évaluation sur le jeu de test...")
        test_results = trainer.test(model, data_loaders["test"])

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

        # Générer et logger les graphiques (Cahier des charges)
        logger.info("Génération des graphiques...")
        generate_training_plots(model, data_loaders, class_names, config)
        
        # Sauvegarder les métriques pour DVC
        save_metrics_for_dvc(final_metrics, test_results)
        
        # Sauvegarder le modèle en production
        logger.info("Sauvegarde du modèle en production...")
        save_model_for_production(trainer, model, config)
        
        # Enregistrer le modèle dans le Model Registry (Cahier des charges)
        logger.info("Enregistrement dans MLflow Model Registry...")
        try:
            mlflow.pytorch.log_model(
                model, 
                "model",
                registered_model_name="plant_disease_model"
            )
        except Exception as e:
            logger.warning(f"Impossible d'enregistrer dans le registry: {e}")

        logger.info("Entraînement terminé avec succès!")

        return model, trainer


def save_model_for_production(trainer, model, config):
    """Sauvegarder le meilleur modèle pour la production."""
    from pathlib import Path
    import shutil
    
    production_dir = Path("models/production")
    production_dir.mkdir(parents=True, exist_ok=True)
    
    # Trouver le meilleur checkpoint
    checkpoint_dir = Path(config["training"]["checkpoint_path"])
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    if checkpoints:
        # Prendre le checkpoint le plus récent (meilleur)
        best_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        dest_path = production_dir / "model.ckpt"
        shutil.copy(best_checkpoint, dest_path)
        logger.info(f"✅ Modèle de production sauvegardé: {dest_path}")
        
        # Sauvegarder aussi les métadonnées
        metadata = {
            "model_type": config["model"]["architecture"],
            "num_classes": config["model"]["num_classes"],
            "image_size": config["data"]["image_size"],
            "checkpoint_source": str(best_checkpoint),
        }
        
        import json
        with open(production_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Métadonnées sauvegardées: {production_dir / 'model_metadata.json'}")
    else:
        logger.warning("⚠️ Aucun checkpoint trouvé pour la production")


def generate_training_plots(model, data_loaders, class_names, config):
    """Générer les graphiques conformes au cahier des charges."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from pathlib import Path
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Évaluer sur le test set pour la matrice de confusion
    model.eval()
    all_preds = []
    all_labels = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in data_loaders["test"]:
            images, labels = batch
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # Logger dans MLflow
    mlflow.log_artifact(str(cm_path))
    logger.info(f"Matrice de confusion sauvegardée: {cm_path}")


def save_metrics_for_dvc(final_metrics, test_results):
    """Sauvegarder les métriques pour le suivi DVC."""
    import json
    from pathlib import Path
    
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    metrics = {}
    for name, value in final_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[name] = float(value.item())
        else:
            metrics[name] = float(value)
    
    # Ajouter les résultats de test
    if test_results:
        for result in test_results:
            for k, v in result.items():
                metrics[f"test_{k}"] = float(v)
    
    with open(metrics_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Métriques sauvegardées pour DVC: metrics/train_metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement des modèles de détection de maladies de plantes"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "vit"],
        default="cnn",
        help="Type de modèle à entraîner",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Chemin vers le dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument("--gpu", type=int, default=None, help="ID du GPU à utiliser")

    args = parser.parse_args()

    # Configuration du GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        train_model(args.model, args.dataset, args.config)
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    main()
