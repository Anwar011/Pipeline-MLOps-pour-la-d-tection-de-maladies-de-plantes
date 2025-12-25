#!/usr/bin/env python3
"""
Script de préparation des données avec versioning DVC.
Conforme au cahier des charges: DataOps avec DVC.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(config_path: str = "config.yaml"):
    """
    Prépare et divise les données pour l'entraînement.
    
    Cette fonction:
    1. Charge les images depuis data/raw
    2. Divise en train/val/test selon la config
    3. Sauvegarde dans data/processed
    4. Génère des statistiques pour le suivi DVC
    """
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    raw_path = Path(data_config["path"])
    processed_path = Path(data_config["processed_path"])
    
    logger.info(f"📂 Préparation des données depuis {raw_path}")
    
    # Vérifier que les données brutes existent
    if not raw_path.exists():
        logger.warning(f"⚠️  Dossier {raw_path} non trouvé. Création d'un dataset de démonstration...")
        create_demo_dataset(raw_path)
    
    # Créer les dossiers de sortie
    for split in ["train", "val", "test"]:
        split_path = processed_path / split
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)
    
    # Collecter les images et labels
    image_paths = []
    labels = []
    class_names = []
    
    # Scanner le dossier des données
    source_path = raw_path
    if (raw_path / "PlantVillage").exists():
        source_path = raw_path / "PlantVillage"
    
    for class_idx, class_name in enumerate(sorted(os.listdir(source_path))):
        class_path = source_path / class_name
        if class_path.is_dir():
            class_names.append(class_name)
            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
    
    logger.info(f"📊 {len(image_paths)} images trouvées dans {len(class_names)} classes")
    
    if len(image_paths) == 0:
        logger.error("❌ Aucune image trouvée!")
        return
    
    # Division des données selon le cahier des charges (70/20/10)
    train_split = data_config["train_split"]
    val_split = data_config["val_split"]
    test_split = data_config["test_split"]
    
    # Premier split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=42
    )
    
    # Deuxième split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=42
    )
    
    # Copier les fichiers vers les dossiers appropriés
    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }
    
    statistics = {
        "total_images": len(image_paths),
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": {}
    }
    
    for split_name, (paths, split_labels) in splits.items():
        split_dir = processed_path / split_name
        class_counts = {}
        
        for img_path, label in zip(paths, split_labels):
            class_name = class_names[label]
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copier l'image
            dest_path = class_dir / Path(img_path).name
            shutil.copy2(img_path, dest_path)
            
            # Compter
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        statistics["splits"][split_name] = {
            "total": len(paths),
            "class_distribution": class_counts
        }
        
        logger.info(f"✅ {split_name}: {len(paths)} images")
    
    # Sauvegarder les statistiques pour DVC
    stats_path = processed_path / "data_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)
    
    # Sauvegarder le mapping des classes
    class_mapping = {str(idx): name for idx, name in enumerate(class_names)}
    mapping_path = Path("data/class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    logger.info(f"📈 Statistiques sauvegardées dans {stats_path}")
    logger.info("✅ Préparation des données terminée!")
    
    return statistics


def create_demo_dataset(raw_path: Path):
    """Crée un dataset de démonstration minimal."""
    import numpy as np
    from PIL import Image
    
    logger.info("🔧 Création d'un dataset de démonstration...")
    
    classes = [
        "Tomato_healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Potato_healthy",
        "Potato_Late_blight"
    ]
    
    raw_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in classes:
        class_dir = raw_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Créer quelques images factices
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"demo_{i}.jpg")
    
    logger.info(f"✅ Dataset de démonstration créé avec {len(classes)} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparer les données pour l'entraînement")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers la configuration")
    args = parser.parse_args()
    
    prepare_data(args.config)
