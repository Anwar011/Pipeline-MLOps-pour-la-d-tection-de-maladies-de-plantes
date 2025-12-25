#!/bin/bash

# ============================================
# Script de Lancement du Monitoring
# Prometheus + Grafana
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

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis pour le monitoring..."

    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé"
        exit 1
    fi

    # Vérifier Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose n'est pas installé"
        exit 1
    fi

    # Vérifier que docker-compose.yml existe
    if [ ! -f "docker/docker-compose.yml" ]; then
        error "Fichier docker/docker-compose.yml manquant"
        exit 1
    fi

    log "Prérequis vérifiés ✓"
}

# Lancement de la stack de monitoring
start_monitoring() {
    log "Lancement de la stack de monitoring..."

    cd docker

    # Lancer les services
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d prometheus grafana
    else
        docker compose up -d prometheus grafana
    fi

    cd ..
    log "Stack de monitoring démarrée ✓"
}

# Arrêt de la stack de monitoring
stop_monitoring() {
    log "Arrêt de la stack de monitoring..."

    cd docker

    if command -v docker-compose &> /dev/null; then
        docker-compose down
    else
        docker compose down
    fi

    cd ..
    log "Stack de monitoring arrêtée ✓"
}

# Redémarrage de la stack
restart_monitoring() {
    log "Redémarrage de la stack de monitoring..."
    stop_monitoring
    sleep 2
    start_monitoring
}

# Affichage du statut
status_monitoring() {
    log "Statut de la stack de monitoring:"

    echo ""
    echo -e "${BLUE}Services Docker:${NC}"
    if command -v docker-compose &> /dev/null; then
        cd docker && docker-compose ps && cd ..
    else
        cd docker && docker compose ps && cd ..
    fi

    echo ""
    echo -e "${BLUE}URLs d'accès:${NC}"
    echo -e "  • Grafana:    ${GREEN}http://localhost:3000${NC} (admin/admin)"
    echo -e "  • Prometheus: ${GREEN}http://localhost:9091${NC}"
    echo -e "  • MLflow:     ${GREEN}http://localhost:5000${NC}"

    echo ""
    echo -e "${BLUE}Commandes de test:${NC}"
    echo -e "  curl http://localhost:9091/api/v1/query?query=api_requests_total"
    echo -e "  curl http://localhost:3000/api/health"
}

# Test des services
test_monitoring() {
    log "Test des services de monitoring..."

    # Attendre que les services démarrent
    sleep 10

    # Test Prometheus
    if curl -s http://localhost:9091/-/healthy | grep -q "Prometheus"; then
        log "Prometheus: ✓"
    else
        error "Prometheus: ✗"
    fi

    # Test Grafana
    if curl -s http://localhost:3000/api/health | grep -q '"database":"ok"'; then
        log "Grafana: ✓"
    else
        error "Grafana: ✗"
    fi

    # Test MLflow
    if curl -s http://localhost:5000/health | grep -q "ok"; then
        log "MLflow: ✓"
    else
        warning "MLflow: ✗ (normal si pas encore démarré)"
    fi

    log "Tests terminés"
}

# Nettoyage des données
clean_monitoring() {
    log "Nettoyage des données de monitoring..."

    warning "Cette action va supprimer toutes les données de monitoring !"
    read -p "Êtes-vous sûr ? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Opération annulée"
        exit 0
    fi

    cd docker

    if command -v docker-compose &> /dev/null; then
        docker-compose down -v
    else
        docker compose down -v
    fi

    cd ..

    # Supprimer les volumes locaux
    rm -rf monitoring/prometheus/data/*
    rm -rf monitoring/grafana/data/*

    log "Données nettoyées ✓"
}

# Logs des services
logs_monitoring() {
    log "Affichage des logs de monitoring..."

    cd docker

    if command -v docker-compose &> /dev/null; then
        docker-compose logs -f prometheus grafana
    else
        docker compose logs -f prometheus grafana
    fi

    cd ..
}

# Fonction principale
main() {
    echo "============================================"
    echo "📊 Monitoring - Détection de Maladies"
    echo "============================================"

    # Vérifier les arguments
    case "${1:-start}" in
        "start")
            check_prerequisites
            start_monitoring
            status_monitoring
            ;;
        "stop")
            stop_monitoring
            ;;
        "restart")
            restart_monitoring
            status_monitoring
            ;;
        "status")
            status_monitoring
            ;;
        "test")
            check_prerequisites
            start_monitoring
            test_monitoring
            ;;
        "logs")
            logs_monitoring
            ;;
        "clean")
            clean_monitoring
            ;;
        *)
            error "Usage: $0 {start|stop|restart|status|test|logs|clean}"
            echo "  start   : Démarrer la stack de monitoring"
            echo "  stop    : Arrêter la stack de monitoring"
            echo "  restart : Redémarrer la stack"
            echo "  status  : Afficher le statut et les URLs"
            echo "  test    : Tester les services"
            echo "  logs    : Afficher les logs"
            echo "  clean   : Nettoyer les données (IRREVERSIBLE)"
            exit 1
            ;;
    esac
}

# Gestion des signaux pour un arrêt propre
trap 'echo -e "\n${YELLOW}Arrêt du monitoring...${NC}"; exit 0' INT TERM

# Lancer la fonction principale
main "$@"