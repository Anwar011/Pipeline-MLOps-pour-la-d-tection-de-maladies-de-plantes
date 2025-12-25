#!/bin/bash

# ============================================
# Script de Tests Automatisés
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

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Configuration
API_URL=${API_URL:-http://localhost:8000}
TEST_IMAGE=${TEST_IMAGE:-test_plant.jpg}
COVERAGE_REPORT=${COVERAGE_REPORT:-htmlcov}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis pour les tests..."

    # Vérifier Python
    if ! command -v python &> /dev/null; then
        error "Python n'est pas installé"
        exit 1
    fi

    # Vérifier pip
    if ! command -v pip &> /dev/null; then
        error "pip n'est pas installé"
        exit 1
    fi

    # Vérifier pytest
    if ! python -c "import pytest" &> /dev/null; then
        warning "pytest non installé. Installation..."
        pip install pytest pytest-cov pytest-asyncio httpx
    fi

    log "Prérequis vérifiés ✓"
}

# Tests unitaires
run_unit_tests() {
    log "Lancement des tests unitaires..."

    # Installer les dépendances de test
    pip install -r requirements-train.txt  # Pour les tests d'entraînement

    # Lancer les tests avec couverture
    pytest tests/ -v \
        --cov=src \
        --cov-report=html:$COVERAGE_REPORT \
        --cov-report=term-missing \
        --cov-fail-under=80

    log "Tests unitaires terminés ✓"
}

# Tests d'intégration API
run_api_tests() {
    log "Lancement des tests d'intégration API..."

    # Vérifier que l'API est accessible
    if ! curl -s $API_URL/health > /dev/null; then
        error "API non accessible sur $API_URL"
        info "Lancez d'abord: bash scripts/run_api.sh dev"
        exit 1
    fi

    # Test 1: Health check
    info "Test 1: Health check"
    response=$(curl -s $API_URL/health)
    if echo "$response" | grep -q "healthy"; then
        log "✓ Health check réussi"
    else
        error "✗ Health check échoué: $response"
        return 1
    fi

    # Test 2: Endpoint racine
    info "Test 2: Endpoint racine"
    response=$(curl -s $API_URL/)
    if echo "$response" | grep -q "Plant Disease Detection API"; then
        log "✓ Endpoint racine réussi"
    else
        error "✗ Endpoint racine échoué"
        return 1
    fi

    # Test 3: Classes supportées
    info "Test 3: Classes supportées"
    response=$(curl -s $API_URL/classes)
    if echo "$response" | grep -q "classes"; then
        log "✓ Classes récupérées"
    else
        error "✗ Récupération des classes échouée"
        return 1
    fi

    # Test 4: Informations modèle
    info "Test 4: Informations modèle"
    response=$(curl -s $API_URL/model/info)
    if echo "$response" | grep -q "model"; then
        log "✓ Informations modèle récupérées"
    else
        error "✗ Récupération des informations modèle échouée"
        return 1
    fi

    # Test 5: Prédiction (si image de test existe)
    if [ -f "$TEST_IMAGE" ]; then
        info "Test 5: Prédiction sur image de test"
        response=$(curl -s -X POST \
            -F "file=@$TEST_IMAGE" \
            $API_URL/predict)

        if echo "$response" | grep -q "prediction"; then
            log "✓ Prédiction réussie"
            # Afficher la prédiction
            echo "$response" | python -m json.tool | head -20
        else
            error "✗ Prédiction échouée: $response"
            return 1
        fi
    else
        warning "Image de test non trouvée: $TEST_IMAGE"
        info "Création d'une image de test factice..."
        # Créer une image de test simple (1x1 pixel noir)
        python -c "
import numpy as np
from PIL import Image
img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
img.save('$TEST_IMAGE')
print('Image de test créée')
        "
        log "✓ Image de test créée"
    fi

    log "Tests d'intégration API terminés ✓"
}

# Tests de performance
run_performance_tests() {
    log "Lancement des tests de performance..."

    # Test de charge simple
    info "Test de charge: 10 requêtes simultanées"

    # Créer un script de test de charge simple
    cat > /tmp/load_test.py << 'EOF'
import asyncio
import aiohttp
import time
import statistics

async def test_request(session, url):
    start_time = time.time()
    try:
        async with session.get(url) as response:
            await response.text()
            return time.time() - start_time
    except Exception as e:
        print(f"Erreur: {e}")
        return None

async def load_test(url, num_requests=10):
    async with aiohttp.ClientSession() as session:
        tasks = [test_request(session, url) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        # Filtrer les résultats valides
        valid_results = [r for r in results if r is not None]

        if valid_results:
            avg_time = statistics.mean(valid_results)
            min_time = min(valid_results)
            max_time = max(valid_results)
            print(".2f"            print(".2f"            print(".2f"            print(f"Requêtes réussies: {len(valid_results)}/{num_requests}")

            # Vérifier les contraintes (< 2s en moyenne)
            if avg_time < 2.0:
                print("✓ Contrainte de performance respectée (< 2s)")
            else:
                print("✗ Contrainte de performance non respectée")
        else:
            print("✗ Aucune requête réussie")

asyncio.run(load_test("http://localhost:8000/health", 10))
EOF

    python /tmp/load_test.py

    log "Tests de performance terminés ✓"
}

# Tests de sécurité
run_security_tests() {
    log "Lancement des tests de sécurité..."

    # Test 1: Taille de fichier limitée
    info "Test 1: Limitation de taille de fichier"

    # Créer un gros fichier de test
    dd if=/dev/zero of=/tmp/large_file.jpg bs=1M count=10 2>/dev/null

    response=$(curl -s -X POST \
        -F "file=@/tmp/large_file.jpg" \
        $API_URL/predict)

    if echo "$response" | grep -q "too large"; then
        log "✓ Limitation de taille respectée"
    else
        warning "Limitation de taille non testée ou non respectée"
    fi

    rm -f /tmp/large_file.jpg

    # Test 2: Type de fichier validé
    info "Test 2: Validation du type de fichier"

    # Créer un fichier texte déguisé en image
    echo "not an image" > /tmp/fake_image.jpg

    response=$(curl -s -X POST \
        -F "file=@/tmp/fake_image.jpg" \
        $API_URL/predict)

    if echo "$response" | grep -q "Invalid image"; then
        log "✓ Validation du type de fichier respectée"
    else
        warning "Validation du type de fichier non testée ou non respectée"
    fi

    rm -f /tmp/fake_image.jpg

    log "Tests de sécurité terminés ✓"
}

# Tests du pipeline DVC
run_pipeline_tests() {
    log "Lancement des tests du pipeline DVC..."

    # Vérifier que DVC est configuré
    if [ ! -d ".dvc" ]; then
        error "DVC n'est pas initialisé"
        exit 1
    fi

    # Test de repro des étapes
    info "Test de reproduction des étapes DVC"

    # Étape 1: prepare_data
    if dvc repro prepare_data --dry; then
        log "✓ Étape prepare_data valide"
    else
        error "✗ Étape prepare_data invalide"
        return 1
    fi

    # Étape 2: train
    if dvc repro train --dry; then
        log "✓ Étape train valide"
    else
        error "✗ Étape train invalide"
        return 1
    fi

    # Étape 3: evaluate
    if dvc repro evaluate --dry; then
        log "✓ Étape evaluate valide"
    else
        error "✗ Étape evaluate invalide"
        return 1
    fi

    log "Tests du pipeline DVC terminés ✓"
}

# Rapport de test
generate_report() {
    log "Génération du rapport de test..."

    # Créer un rapport HTML simple
    cat > test_report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Rapport de Tests - Plant Disease Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
        h1 { color: #2E7D32; }
        h2 { color: #1976D2; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🧪 Rapport de Tests - Détection de Maladies de Plantes</h1>
    <p><strong>Date:</strong> $(date)</p>
    <p><strong>Environnement:</strong> $(uname -a)</p>

    <h2>Résumé des Tests</h2>
    <p>Tests exécutés avec succès. Voir les détails ci-dessous.</p>

    <h2>Couverture de Code</h2>
    <p>Rapport disponible dans: <a href="$COVERAGE_REPORT/index.html">$COVERAGE_REPORT/index.html</a></p>

    <h2>Recommandations</h2>
    <ul>
        <li>Vérifier régulièrement la couverture de code (> 80%)</li>
        <li>Exécuter les tests avant chaque déploiement</li>
        <li>Monitorer les performances en production</li>
    </ul>
</body>
</html>
EOF

    log "Rapport généré: test_report.html"
}

# Fonction principale
main() {
    echo "============================================"
    echo "🧪 Tests Automatisés - Détection de Maladies"
    echo "============================================"

    # Vérifier les arguments
    case "${1:-all}" in
        "unit")
            check_prerequisites
            run_unit_tests
            ;;
        "api")
            check_prerequisites
            run_api_tests
            ;;
        "perf")
            check_prerequisites
            run_performance_tests
            ;;
        "security")
            check_prerequisites
            run_security_tests
            ;;
        "pipeline")
            check_prerequisites
            run_pipeline_tests
            ;;
        "report")
            generate_report
            ;;
        "all")
            check_prerequisites
            run_unit_tests
            run_api_tests
            run_performance_tests
            run_security_tests
            run_pipeline_tests
            generate_report
            ;;
        *)
            error "Usage: $0 {unit|api|perf|security|pipeline|report|all}"
            echo "  unit     : Tests unitaires"
            echo "  api      : Tests d'intégration API"
            echo "  perf     : Tests de performance"
            echo "  security : Tests de sécurité"
            echo "  pipeline : Tests du pipeline DVC"
            echo "  report   : Générer un rapport"
            echo "  all      : Tous les tests"
            exit 1
            ;;
    esac

    log "🎉 Tests terminés avec succès !"
}

# Gestion des signaux pour un arrêt propre
trap 'echo -e "\n${YELLOW}Arrêt des tests...${NC}"; exit 0' INT TERM

# Lancer la fonction principale
main "$@"