"""
Module de prétraitement des données pour la détection de maladies de plantes.
Inclut le chargement, l'augmentation et la préparation des datasets.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """Dataset personnalisé pour les images de maladies de plantes."""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): Liste des chemins vers les images
            labels (list): Liste des labels correspondants
            transform (callable, optional): Transformations à appliquer
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Charger l'image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Appliquer les transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        label = self.labels[idx]
        return image, label

class DataPreprocessor:
    """Classe pour le prétraitement des données."""

    def __init__(self, config_path="config.yaml"):
        """Initialiser le préprocesseur avec la configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.setup_directories()

    def setup_directories(self):
        """Créer les répertoires nécessaires."""
        Path(self.data_config['processed_path']).mkdir(parents=True, exist_ok=True)
        Path(self.data_config['path']).mkdir(parents=True, exist_ok=True)

    def get_data_augmentation(self, is_training=True):
        """Définir les transformations d'augmentation des données."""
        if is_training:
            return A.Compose([
                A.Resize(height=self.data_config['image_size'][0],
                        width=self.data_config['image_size'][1]),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                         contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=self.data_config['image_size'][0],
                        width=self.data_config['image_size'][1]),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def load_dataset_info(self, dataset_path):
        """
        Charger les informations du dataset PlantVillage.
        Args:
            dataset_path (str): Chemin vers le dataset
        Returns:
            tuple: (image_paths, labels, class_names)
        """
        image_paths = []
        labels = []
        class_names = []

        # Parcourir les dossiers de classes
        for class_idx, class_name in enumerate(sorted(os.listdir(dataset_path))):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                class_names.append(class_name)
                logger.info(f"Chargement de la classe: {class_name}")

                # Parcourir les images de la classe
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_path, image_file))
                        labels.append(class_idx)

        logger.info(f"Dataset chargé: {len(image_paths)} images, {len(class_names)} classes")
        return image_paths, labels, class_names

    def split_dataset(self, image_paths, labels):
        """
        Diviser le dataset en ensembles d'entraînement, validation et test.
        """
        # Premier split: train + (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(self.data_config['val_split'] + self.data_config['test_split']),
            stratify=labels,
            random_state=42
        )

        # Deuxième split: val et test
        val_size = self.data_config['val_split'] / (self.data_config['val_split'] + self.data_config['test_split'])
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=42
        )

        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }

    def create_data_loaders(self, dataset_path):
        """
        Créer les DataLoaders pour l'entraînement.
        Args:
            dataset_path (str): Chemin vers le dataset
        Returns:
            dict: Dictionnaire contenant les DataLoaders et informations sur les classes
        """
        # Charger les données
        image_paths, labels, class_names = self.load_dataset_info(dataset_path)

        # Diviser le dataset
        splits = self.split_dataset(image_paths, labels)

        # Créer les datasets
        data_loaders = {}
        for split_name, (paths, split_labels) in splits.items():
            if split_name == 'train':
                transform = self.get_data_augmentation(is_training=True)
            else:
                transform = self.get_data_augmentation(is_training=False)

            dataset = PlantDiseaseDataset(paths, split_labels, transform=transform)

            data_loaders[split_name] = DataLoader(
                dataset,
                batch_size=self.data_config['batch_size'],
                shuffle=(split_name == 'train'),
                num_workers=self.data_config['num_workers'],
                pin_memory=True
            )

        return {
            'data_loaders': data_loaders,
            'class_names': class_names,
            'num_classes': len(class_names)
        }

    def save_class_mapping(self, class_names, output_path="data/class_mapping.json"):
        """Sauvegarder le mapping des classes."""
        class_mapping = {idx: name for idx, name in enumerate(class_names)}
        with open(output_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        logger.info(f"Mapping des classes sauvegardé dans {output_path}")

def main():
    """Fonction principale pour tester le prétraitement."""
    preprocessor = DataPreprocessor()

    # Chemin vers le dataset (à adapter selon votre structure)
    dataset_path = "data/raw/PlantVillage"

    if os.path.exists(dataset_path):
        result = preprocessor.create_data_loaders(dataset_path)

        # Afficher les statistiques
        for split_name, loader in result['data_loaders'].items():
            logger.info(f"{split_name}: {len(loader.dataset)} images")

        logger.info(f"Classes: {result['class_names']}")
        preprocessor.save_class_mapping(result['class_names'])
    else:
        logger.warning(f"Dataset non trouvé à {dataset_path}")
        logger.info("Veuillez télécharger le dataset PlantVillage depuis Kaggle")

if __name__ == "__main__":
    main()
