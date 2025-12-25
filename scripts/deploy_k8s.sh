#!/bin/bash

# ============================================
# Script de Déploiement Kubernetes
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
NAMESPACE=${K8S_NAMESPACE:-plant-disease}
TIMEOUT=${K8S_TIMEOUT:-300}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis Kubernetes..."

    # Vérifier kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl n'est pas installé"
        exit 1
    fi

    # Vérifier la connexion au cluster
    if ! kubectl cluster-info &> /dev/null; then
        error "Impossible de se connecter au cluster Kubernetes"
        exit 1
    fi

    # Vérifier Docker (pour les images)
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé"
        exit 1
    fi

    log "Prérequis vérifiés ✓"
}

# Création du namespace
create_namespace() {
    log "Création du namespace: $NAMESPACE"

    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        kubectl create namespace $NAMESPACE
        log "Namespace créé"
    else
        log "Namespace existe déjà"
    fi
}

# Construction et push des images
build_and_push_images() {
    log "Construction et push des images Docker..."

    # Configuration du registry (local pour la démo)
    REGISTRY=${DOCKER_REGISTRY:-localhost:5000}
    TAG=${DOCKER_TAG:-latest}

    # Image d'inférence
    IMAGE_NAME="$REGISTRY/plant-disease-mlops:inference-$TAG"

    info "Construction de l'image: $IMAGE_NAME"
    docker build -f docker/Dockerfile.inference -t $IMAGE_NAME .

    info "Push de l'image: $IMAGE_NAME"
    docker push $IMAGE_NAME

    # Mettre à jour les manifests avec la nouvelle image
    sed -i "s|image:.*|image: $IMAGE_NAME|g" k8s/deployment.yaml

    log "Images construites et poussées ✓"
}

# Déploiement des ressources
deploy_resources() {
    log "Déploiement des ressources Kubernetes..."

    # Créer le namespace
    create_namespace

    # Storage
    info "Déploiement du stockage..."
    kubectl apply -f k8s/storage.yaml -n $NAMESPACE

    # ConfigMaps et Secrets si nécessaire
    # kubectl apply -f k8s/configmap.yaml -n $NAMESPACE

    # Service
    info "Déploiement du service..."
    kubectl apply -f k8s/service.yaml -n $NAMESPACE

    # Deployment
    info "Déploiement de l'application..."
    kubectl apply -f k8s/deployment.yaml -n $NAMESPACE

    # HPA
    info "Déploiement de l'HPA..."
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE

    log "Ressources déployées ✓"
}

# Attendre que le déploiement soit prêt
wait_for_deployment() {
    log "Attente de la disponibilité du déploiement..."

    kubectl wait --for=condition=available --timeout=${TIMEOUT}s deployment/plant-disease-api -n $NAMESPACE

    log "Déploiement prêt ✓"
}

# Affichage du statut
show_status() {
    log "Statut du déploiement:"

    echo ""
    echo -e "${BLUE}Pods:${NC}"
    kubectl get pods -n $NAMESPACE

    echo ""
    echo -e "${BLUE}Services:${NC}"
    kubectl get services -n $NAMESPACE

    echo ""
    echo -e "${BLUE}HPA:${NC}"
    kubectl get hpa -n $NAMESPACE

    echo ""
    echo -e "${BLUE}URLs d'accès:${NC}"
    EXTERNAL_IP=$(kubectl get svc plant-disease-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    echo -e "  • API:        ${GREEN}http://$EXTERNAL_IP${NC}"
    echo -e "  • Docs:       ${GREEN}http://$EXTERNAL_IP/docs${NC}"
    echo -e "  • Health:     ${GREEN}http://$EXTERNAL_IP/health${NC}"
    echo -e "  • Metrics:    ${GREEN}http://$EXTERNAL_IP:9090/metrics${NC}"
}

# Test du déploiement
test_deployment() {
    log "Test du déploiement..."

    # Obtenir l'IP du service
    EXTERNAL_IP=$(kubectl get svc plant-disease-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")

    # Attendre que le service soit prêt
    sleep 10

    # Test health check
    if curl -s http://$EXTERNAL_IP/health | grep -q "healthy"; then
        log "Health check: ✓"
    else
        error "Health check: ✗"
        return 1
    fi

    # Test endpoint racine
    if curl -s http://$EXTERNAL_IP/ | grep -q "Plant Disease Detection API"; then
        log "API endpoint: ✓"
    else
        error "API endpoint: ✗"
        return 1
    fi

    log "Tests terminés ✓"
}

# Mise à jour du déploiement
update_deployment() {
    log "Mise à jour du déploiement..."

    # Rebuild et push des images
    build_and_push_images

    # Redéployer
    kubectl rollout restart deployment/plant-disease-api -n $NAMESPACE

    # Attendre la mise à jour
    kubectl rollout status deployment/plant-disease-api -n $NAMESPACE --timeout=${TIMEOUT}s

    log "Déploiement mis à jour ✓"
}

# Nettoyage du déploiement
cleanup_deployment() {
    log "Nettoyage du déploiement..."

    warning "Cette action va supprimer toutes les ressources Kubernetes !"
    read -p "Êtes-vous sûr ? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Opération annulée"
        exit 0
    fi

    # Supprimer les ressources
    kubectl delete -f k8s/ --ignore-not-found=true -n $NAMESPACE

    # Supprimer le namespace
    kubectl delete namespace $NAMESPACE --ignore-not-found=true

    log "Nettoyage terminé ✓"
}

# Logs des pods
show_logs() {
    log "Affichage des logs des pods..."

    kubectl logs -f deployment/plant-disease-api -n $NAMESPACE
}

# Fonction principale
main() {
    echo "============================================"
    echo "☸️  Déploiement Kubernetes - Détection de Maladies"
    echo "============================================"

    # Vérifier les arguments
    case "${1:-deploy}" in
        "check")
            check_prerequisites
            ;;
        "build")
            check_prerequisites
            build_and_push_images
            ;;
        "deploy")
            check_prerequisites
            build_and_push_images
            deploy_resources
            wait_for_deployment
            show_status
            ;;
        "test")
            check_prerequisites
            test_deployment
            ;;
        "update")
            check_prerequisites
            update_deployment
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        *)
            error "Usage: $0 {check|build|deploy|test|update|status|logs|cleanup}"
            echo "  check   : Vérifier les prérequis"
            echo "  build   : Construire et pousser les images"
            echo "  deploy  : Déploiement complet"
            echo "  test    : Tester le déploiement"
            echo "  update  : Mettre à jour le déploiement"
            echo "  status  : Afficher le statut"
            echo "  logs    : Afficher les logs"
            echo "  cleanup : Nettoyer le déploiement (IRREVERSIBLE)"
            exit 1
            ;;
    esac
}

# Gestion des signaux pour un arrêt propre
trap 'echo -e "\n${YELLOW}Arrêt du déploiement...${NC}"; exit 0' INT TERM

# Lancer la fonction principale
main "$@"