#!/usr/bin/env python3
"""
Script de test simple pour vérifier que l'API fonctionne.
"""

import sys
import os
sys.path.append('src')

def test_api():
    """Test basique de l'API."""
    print("🧪 Test de l'API d'inférence...")

    try:
        from models import create_model
        import json

        # Tester le chargement du modèle
        model = create_model('cnn')
        print("✅ Modèle CNN créé avec succès")

        # Tester le mapping des classes
        with open('data/class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        print(f"✅ Mapping des classes chargé: {len(class_mapping)} classes")

        print("🎉 Infrastructure fonctionnelle!")
        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    test_api()
