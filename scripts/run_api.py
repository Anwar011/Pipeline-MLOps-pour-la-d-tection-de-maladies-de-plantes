#!/usr/bin/env python3
"""
Script pour lancer l'API d'inférence de détection de maladies de plantes.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api import main as api_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Vérifie que tous les prérequis sont satisfaits."""
    # Vérifier que le modèle existe
    model_path = "models/production/model.ckpt"
    if not os.path.exists(model_path):
        logger.warning(f"⚠️  Modèle non trouvé: {model_path}")
        logger.info("💡 Entraînez d'abord un modèle avec: python scripts/train_pipeline.py --dataset <path>")
        return False

    # Vérifier que le mapping des classes existe
    mapping_path = "data/class_mapping.json"
    if not os.path.exists(mapping_path):
        logger.warning(f"⚠️  Mapping des classes non trouvé: {mapping_path}")
        logger.info("💡 Le mapping sera créé automatiquement lors du premier entraînement")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Lancer l'API d'inférence")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host pour l'API"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port pour l'API"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers la configuration"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Ignorer les vérifications de prérequis"
    )

    args = parser.parse_args()

    logger.info("🚀 Démarrage de l'API de détection de maladies de plantes")

    # Vérifications de prérequis
    if not args.skip_checks and not check_requirements():
        logger.error("❌ Prérequis non satisfaits. Utilisez --skip-checks pour forcer le démarrage.")
        sys.exit(1)

    # Modifier les variables d'environnement si nécessaire
    os.environ.setdefault("API_HOST", args.host)
    os.environ.setdefault("API_PORT", str(args.port))

    try:
        # Lancer l'API
        logger.info(f"🌐 API accessible sur http://{args.host}:{args.port}")
        logger.info("📖 Documentation disponible sur http://{args.host}:{args.port}/docs")
        logger.info("🩺 Health check sur http://{args.host}:{args.port}/health")
        logger.info("📊 Métriques Prometheus sur http://{args.host}:{args.port}/metrics")

        api_main()

    except KeyboardInterrupt:
        logger.info("🛑 Arrêt de l'API demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de l'API: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
