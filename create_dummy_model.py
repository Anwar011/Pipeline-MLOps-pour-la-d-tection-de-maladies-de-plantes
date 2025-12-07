#!/usr/bin/env python3
"""
Créer un modèle factice pour tester l'API.
"""

import torch
import sys
import os
sys.path.append('src')

from models import create_model

def create_dummy_model():
    """Créer et sauvegarder un modèle factice."""
    print("🤖 Création d'un modèle factice...")

    try:
        # Créer le modèle
        model = create_model('cnn')

        # Créer le répertoire si nécessaire
        os.makedirs('models/production', exist_ok=True)

        # Sauvegarder le modèle factice
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': 0,
            'val_acc': 0.5,
            'config': {'num_classes': 15}
        }

        torch.save(checkpoint, 'models/production/model.ckpt')
        print("✅ Modèle factice créé: models/production/model.ckpt")

        # Créer le mapping des classes
        class_names = [
            "Pepper__bell___Bacterial_spot",
            "Pepper__bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            "Tomato_healthy"
        ]

        class_mapping = {i: name for i, name in enumerate(class_names)}

        import json
        with open('data/class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)

        print("✅ Mapping des classes créé: data/class_mapping.json")

        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    create_dummy_model()
