#!/bin/bash
# Script de démarrage rapide pour le pipeline automatisé

set -e

echo "🚀 Démarrage du pipeline automatisé MLOps"
echo "=========================================="
echo ""

# Vérifier les prérequis
echo "📋 Vérification des prérequis..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi
echo "✅ Python 3 trouvé"

# Vérifier DVC
if ! command -v dvc &> /dev/null; then
    echo "⚠️  DVC n'est pas installé. Installation..."
    pip install dvc
fi
echo "✅ DVC trouvé"

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi
echo "✅ Docker trouvé"

# Vérifier Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi
echo "✅ Docker Compose trouvé"

echo ""
echo "🎯 Choisissez une option:"
echo "1. Exécuter le pipeline une fois"
echo "2. Démarrer la surveillance continue"
echo "3. Vérifier les changements DVC uniquement"
echo ""

read -p "Votre choix (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Exécution du pipeline..."
        python3 scripts/run_automated_pipeline.py
        ;;
    2)
        echo ""
        echo "👀 Démarrage de la surveillance continue..."
        echo "💡 Appuyez sur Ctrl+C pour arrêter"
        python3 scripts/watch_and_trigger.py
        ;;
    3)
        echo ""
        echo "🔍 Vérification des changements DVC..."
        python3 scripts/monitor_dvc_changes.py
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

