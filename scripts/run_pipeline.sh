#!/bin/bash

# ============================================
# Script Pipeline MLOps Complet
# Détection de Maladies de Plantes
# ============================================

set -e  # Arrêter le script en cas d'erreur

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis..."

    # Vérifier Python
    if ! command -v python3 &> /dev/null; then
        error "Python n'est pas installé"
        exit 1
    fi

    # Vérifier pip
    if ! command -v pip3 &> /dev/null; then
        error "pip n'est pas installé"
        exit 1
    fi

    # Vérifier DVC
    if ! command -v dvc &> /dev/null; then
        error "DVC n'est pas installé. Installez-le avec: pip install dvc"
        exit 1
    fi

    # Vérifier MLflow
    if ! python -c "import mlflow" &> /dev/null; then
        error "MLflow n'est pas installé. Installez-le avec: pip install mlflow"
        exit 1
    fi

    log "Prérequis vérifiés ✓"
}

# Installation des dépendances
install_dependencies() {
    log "Installation des dépendances..."

    # Installer les dépendances de base
    pip install -r requirements.txt

    # Installer les dépendances d'entraînement
    pip install -r requirements-train.txt

    log "Dépendances installées ✓"
}

# Initialisation DVC
init_dvc() {
    log "Initialisation DVC..."

    # Initialiser DVC si pas déjà fait
    if [ ! -d ".dvc" ]; then
        dvc init
        log "DVC initialisé"
    else
        log "DVC déjà initialisé"
    fi

    # Configurer le stockage distant (local pour la démo)
    if [ ! -f ".dvc/config" ] || ! grep -q "core.remote" .dvc/config; then
        dvc remote add -d myremote ./dvc_storage
        log "Remote DVC configuré"
    fi
}

# Pipeline DVC complet
run_pipeline() {
    log "Lancement du pipeline DVC complet..."

    # Étape 1: Préparation des données
    info "Étape 1/5: Préparation des données"
    dvc repro prepare_data

    # Étape 2: Entraînement
    info "Étape 2/5: Entraînement du modèle"
    dvc repro train

    # Étape 3: Évaluation
    info "Étape 3/5: Évaluation du modèle"
    dvc repro evaluate

    # Étape 4: Export du modèle
    info "Étape 4/5: Export du modèle"
    dvc repro export_model

    # Étape 5: Analyse de dérive
    info "Étape 5/5: Analyse de dérive"
    dvc repro drift_analysis

    log "Pipeline DVC terminé ✓"
}

# Tests unitaires
run_tests() {
    log "Lancement des tests unitaires..."

    # Installer les dépendances de test
    pip install pytest pytest-cov

    # Lancer les tests
    pytest tests/ -v --cov=src --cov-report=html

    log "Tests terminés ✓"
}

# Construction des images Docker
build_docker() {
    log "Construction des images Docker..."

    # Image d'entraînement
    docker build -f docker/Dockerfile.train -t plant-disease-mlops:train .
    log "Image d'entraînement construite"

    # Image d'inférence
    docker build -f docker/Dockerfile.inference -t plant-disease-mlops:inference .
    log "Image d'inférence construite"

    # Image optimisée
    docker build -f docker/Dockerfile.inference.optimized -t plant-disease-mlops:inference-optimized .
    log "Image optimisée construite"

    log "Images Docker construites ✓"
}

# Fonction principale
main() {
    echo "============================================"
    echo "🚀 Pipeline MLOps - Détection de Maladies"
    echo "============================================"

    # Vérifier les arguments
    case "${1:-all}" in
        "check")
            check_prerequisites
            ;;
        "install")
            check_prerequisites
            install_dependencies
            ;;
        "init")
            check_prerequisites
            install_dependencies
            init_dvc
            ;;
        "pipeline")
            check_prerequisites
            install_dependencies
            init_dvc
            run_pipeline
            ;;
        "test")
            check_prerequisites
            install_dependencies
            run_tests
            ;;
        "docker")
            build_docker
            ;;
        "all")
            check_prerequisites
            install_dependencies
            init_dvc
            run_pipeline
            run_tests
            build_docker
            ;;
        *)
            error "Usage: $0 {check|install|init|pipeline|test|docker|all}"
            exit 1
            ;;
    esac

    log "🎉 Opération terminée avec succès !"
}

# Lancer la fonction principale
main "$@"