#!/bin/bash

# ============================================
# Script de Lancement de l'API FastAPI
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

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Configuration
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-4}
MODEL_PATH=${MODEL_PATH:-models/best_model.ckpt}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis pour l'API..."

    # Vérifier Python
    if ! command -v python &> /dev/null; then
        error "Python n'est pas installé"
        exit 1
    fi

    # Vérifier le modèle
    if [ ! -f "$MODEL_PATH" ]; then
        warning "Modèle non trouvé: $MODEL_PATH"
        info "Lancement de l'entraînement automatique..."

        # Lancer l'entraînement si le modèle n'existe pas
        bash scripts/run_pipeline.sh pipeline
    fi

    # Vérifier les dépendances
    if ! python -c "import fastapi, uvicorn, torch" &> /dev/null; then
        error "Dépendances manquantes. Installez-les avec: pip install -r requirements-inference.txt"
        exit 1
    fi

    log "Prérequis vérifiés ✓"
}

# Lancement de l'API en mode développement
run_dev() {
    log "Lancement de l'API en mode développement..."
    info "URL: http://$HOST:$PORT"
    info "Documentation: http://$HOST:$PORT/docs"
    info "Health check: http://$HOST:$PORT/health"

    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    uvicorn src.api:app --host $HOST --port $PORT --reload --log-level info
}

# Lancement de l'API en mode production
run_prod() {
    log "Lancement de l'API en mode production..."
    info "Workers: $WORKERS"
    info "URL: http://$HOST:$PORT"
    info "Documentation: http://$HOST:$PORT/docs"

    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    uvicorn src.api:app --host $HOST --port $PORT --workers $WORKERS --log-level warning
}

# Lancement avec Gunicorn (recommandé pour production)
run_gunicorn() {
    log "Lancement de l'API avec Gunicorn..."
    info "Workers: $WORKERS"
    info "URL: http://$HOST:$PORT"

    # Vérifier Gunicorn
    if ! command -v gunicorn &> /dev/null; then
        warning "Gunicorn non installé. Installation..."
        pip install gunicorn
    fi

    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    gunicorn src.api:app \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --log-level warning \
        --access-logfile - \
        --error-logfile -
}

# Test de l'API
test_api() {
    log "Test de l'API..."

    # Attendre que l'API démarre
    sleep 3

    # Test health check
    if curl -s http://localhost:$PORT/health > /dev/null; then
        log "Health check: ✓"
    else
        error "Health check: ✗"
        return 1
    fi

    # Test endpoint racine
    if curl -s http://localhost:$PORT/ | grep -q "Plant Disease Detection API"; then
        log "Endpoint racine: ✓"
    else
        error "Endpoint racine: ✗"
        return 1
    fi

    # Test classes
    if curl -s http://localhost:$PORT/classes | grep -q "classes"; then
        log "Endpoint classes: ✓"
    else
        error "Endpoint classes: ✗"
        return 1
    fi

    log "Tests API terminés ✓"
}

# Fonction principale
main() {
    echo "============================================"
    echo "🚀 Lancement API - Détection de Maladies"
    echo "============================================"

    # Vérifier les arguments
    case "${1:-dev}" in
        "dev")
            check_prerequisites
            run_dev
            ;;
        "prod")
            check_prerequisites
            run_prod
            ;;
        "gunicorn")
            check_prerequisites
            run_gunicorn
            ;;
        "test")
            check_prerequisites
            # Lancer l'API en arrière-plan pour les tests
            run_dev &
            API_PID=$!
            sleep 5
            test_api
            kill $API_PID 2>/dev/null || true
            ;;
        *)
            error "Usage: $0 {dev|prod|gunicorn|test}"
            echo "  dev      : Mode développement (avec rechargement automatique)"
            echo "  prod     : Mode production (uvicorn)"
            echo "  gunicorn : Mode production (gunicorn recommandé)"
            echo "  test     : Test automatique de l'API"
            exit 1
            ;;
    esac
}

# Gestion des signaux pour un arrêt propre
trap 'echo -e "\n${YELLOW}Arrêt de l'\''API...${NC}"; exit 0' INT TERM

# Lancer la fonction principale
main "$@"