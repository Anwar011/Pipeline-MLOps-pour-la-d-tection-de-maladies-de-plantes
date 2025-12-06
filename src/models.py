"""
Module contenant les architectures de modèles pour la détection de maladies de plantes.
Implémente CNN (ResNet, EfficientNet) et Vision Transformer.
"""

import torch
import torch.nn as nn
import torchvision.models as models
# Temporarily disable ViT due to transformers compatibility issues
# from transformers.models.vit.modeling_vit import ViTModel
# from transformers.models.vit.configuration_vit import ViTConfig
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Accuracy
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PlantDiseaseCNN(pl.LightningModule):
    """Modèle CNN pour la classification des maladies de plantes."""

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.save_hyperparameters()

        # Charger la configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_config = config['model']
        self.training_config = config['training']

        # Nombre de classes
        self.num_classes = self.model_config['num_classes']

        # Créer le modèle de base
        self.backbone = self._create_backbone()
        self.classifier = nn.Linear(self._get_backbone_output_size(), self.num_classes)

        # Fonctions de perte
        self.criterion = nn.CrossEntropyLoss()

        # Métriques
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _create_backbone(self):
        """Créer le backbone CNN selon la configuration."""
        architecture = self.model_config['architecture']

        if architecture == 'resnet50':
            backbone = models.resnet50(pretrained=self.model_config['pretrained'])
            if self.model_config['freeze_backbone']:
                for param in backbone.parameters():
                    param.requires_grad = False
            # Remplacer la dernière couche
            backbone.fc = nn.Identity()

        elif architecture == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=self.model_config['pretrained'])
            if self.model_config['freeze_backbone']:
                for param in backbone.parameters():
                    param.requires_grad = False
            backbone.classifier = nn.Identity()

        elif architecture == 'vgg16':
            backbone = models.vgg16(pretrained=self.model_config['pretrained'])
            if self.model_config['freeze_backbone']:
                for param in backbone.parameters():
                    param.requires_grad = False
            backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])

        else:
            raise ValueError(f"Architecture {architecture} non supportée")

        return backbone

    def _get_backbone_output_size(self):
        """Obtenir la taille de sortie du backbone."""
        architecture = self.model_config['architecture']

        if architecture == 'resnet50':
            return 2048
        elif architecture == 'efficientnet_b0':
            return 1280
        elif architecture == 'vgg16':
            return 4096
        else:
            raise ValueError(f"Architecture {architecture} non supportée")

    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Métriques
        self.train_accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Métriques
        self.val_accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Métriques
        self.test_accuracy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configurer l'optimiseur et le scheduler."""
        if self.training_config['optimizer'] == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = SGD(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                momentum=0.9,
                weight_decay=self.training_config['weight_decay']
            )
        else:
            raise ValueError(f"Optimiseur {self.training_config['optimizer']} non supporté")

        if self.training_config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.training_config['epochs'])
        elif self.training_config['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class PlantDiseaseViT(pl.LightningModule):
    """Modèle Vision Transformer pour la classification des maladies de plantes."""

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.save_hyperparameters()

        # Charger la configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_config = config['model']
        self.training_config = config['training']
        self.num_classes = self.model_config['num_classes']

        # Configuration ViT - Temporarily disabled due to transformers compatibility issues
        # self.vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        # self.vit_config.num_labels = self.num_classes

        # Créer le modèle ViT - Temporarily disabled
        # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', config=self.vit_config)
        raise NotImplementedError("ViT model temporarily disabled due to transformers compatibility issues")
        self.classifier = nn.Linear(self.vit_config.hidden_size, self.num_classes)

        # Geler le backbone si configuré
        if self.model_config.get('freeze_backbone', False):
            for param in self.vit.parameters():
                param.requires_grad = False

        # Fonctions de perte
        self.criterion = nn.CrossEntropyLoss()

        # Métriques
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        """Forward pass."""
        outputs = self.vit(x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.test_accuracy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configurer l'optimiseur et le scheduler."""
        if self.training_config['optimizer'] == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        else:
            optimizer = Adam(self.parameters(), lr=self.training_config['learning_rate'])

        if self.training_config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.training_config['epochs'])
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def create_model(model_type="cnn", config_path="config.yaml"):
    """
    Fonction factory pour créer un modèle.
    Args:
        model_type (str): Type de modèle ('cnn' ou 'vit')
        config_path (str): Chemin vers le fichier de configuration
    Returns:
        LightningModule: Le modèle créé
    """
    if model_type.lower() == 'cnn':
        return PlantDiseaseCNN(config_path)
    elif model_type.lower() == 'vit':
        raise NotImplementedError("ViT model temporarily disabled due to transformers compatibility issues")
    else:
        raise ValueError(f"Type de modèle {model_type} non supporté")

# Callbacks personnalisés
class ModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """Callback personnalisé pour sauvegarder les meilleurs modèles."""

    def __init__(self, **kwargs):
        super().__init__(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            **kwargs
        )

class EarlyStoppingCallback(pl.callbacks.EarlyStopping):
    """Callback d'arrêt précoce."""

    def __init__(self, patience=10, **kwargs):
        super().__init__(
            monitor='val_loss',
            patience=patience,
            mode='min',
            **kwargs
        )

def get_training_callbacks(config_path="config.yaml"):
    """Obtenir les callbacks pour l'entraînement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    training_config = config['training']
    checkpoint_path = training_config['checkpoint_path']
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpointCallback(
            dirpath=checkpoint_path,
            filename='{epoch}-{val_acc:.2f}'
        ),
        EarlyStoppingCallback(
            patience=training_config['early_stopping_patience']
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    return callbacks
