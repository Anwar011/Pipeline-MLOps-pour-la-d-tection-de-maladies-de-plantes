#!/bin/bash

# ============================================
# Script Principal - Pipeline MLOps
# Détection de Maladies de Plantes
# ============================================

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Fonction d'affichage du menu
show_menu() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           🌱 Pipeline MLOps - Détection de Maladies 🌱          ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${WHITE}Scripts disponibles:${NC}"
    echo ""

    echo -e "${GREEN}🚀 Pipeline Complet:${NC}"
    echo -e "  ${YELLOW}./run_pipeline.sh${NC} [check|install|init|pipeline|test|docker|all]"
    echo -e "    • check   : Vérifier les prérequis"
    echo -e "    • install : Installer les dépendances"
    echo -e "    • init    : Initialiser DVC"
    echo -e "    • pipeline: Exécuter le pipeline complet"
    echo -e "    • test    : Lancer les tests"
    echo -e "    • docker  : Construire les images"
    echo -e "    • all     : Tout exécuter"
    echo ""

    echo -e "${GREEN}🌐 API FastAPI:${NC}"
    echo -e "  ${YELLOW}./run_api.sh${NC} [dev|prod|gunicorn|test]"
    echo -e "    • dev      : Mode développement (rechargement auto)"
    echo -e "    • prod     : Mode production (uvicorn)"
    echo -e "    • gunicorn : Mode production (recommandé)"
    echo -e "    • test     : Tester l'API automatiquement"
    echo ""

    echo -e "${GREEN}📊 Monitoring:${NC}"
    echo -e "  ${YELLOW}./run_monitoring.sh${NC} [start|stop|restart|status|test|logs|clean]"
    echo -e "    • start   : Démarrer Prometheus + Grafana"
    echo -e "    • stop    : Arrêter la stack"
    echo -e "    • restart : Redémarrer"
    echo -e "    • status  : Afficher le statut et URLs"
    echo -e "    • test    : Tester les services"
    echo -e "    • logs    : Afficher les logs"
    echo -e "    • clean   : Nettoyer les données"
    echo ""

    echo -e "${GREEN}☸️  Kubernetes:${NC}"
    echo -e "  ${YELLOW}./deploy_k8s.sh${NC} [check|build|deploy|test|update|status|logs|cleanup]"
    echo -e "    • check   : Vérifier les prérequis"
    echo -e "    • build   : Construire les images"
    echo -e "    • deploy  : Déploiement complet"
    echo -e "    • test    : Tester le déploiement"
    echo -e "    • update  : Mettre à jour"
    echo -e "    • status  : Afficher le statut"
    echo -e "    • logs    : Afficher les logs"
    echo -e "    • cleanup : Nettoyer (IRREVERSIBLE)"
    echo ""

    echo -e "${GREEN}🧪 Tests:${NC}"
    echo -e "  ${YELLOW}./run_tests.sh${NC} [unit|api|perf|security|pipeline|report|all]"
    echo -e "    • unit     : Tests unitaires"
    echo -e "    • api      : Tests d'intégration API"
    echo -e "    • perf     : Tests de performance"
    echo -e "    • security : Tests de sécurité"
    echo -e "    • pipeline : Tests du pipeline DVC"
    echo -e "    • report   : Générer un rapport"
    echo -e "    • all      : Tous les tests"
    echo ""

    echo -e "${GREEN}🎪 Démonstration:${NC}"
    echo -e "  ${YELLOW}./demo_presentation.sh${NC}"
    echo -e "    • Présentation interactive du projet"
    echo ""

    echo -e "${MAGENTA}URLs importantes:${NC}"
    echo -e "  • API Local:     ${CYAN}http://localhost:8000${NC}"
    echo -e "  • Docs API:      ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  • Grafana:       ${CYAN}http://localhost:3000${NC} (admin/admin)"
    echo -e "  • Prometheus:    ${CYAN}http://localhost:9091${NC}"
    echo -e "  • MLflow:        ${CYAN}http://localhost:5000${NC}"
    echo ""

    echo -e "${YELLOW}Workflow recommandé:${NC}"
    echo -e "  1. ${WHITE}./run_pipeline.sh all${NC}     # Pipeline complet"
    echo -e "  2. ${WHITE}./run_api.sh dev${NC}          # Lancer l'API"
    echo -e "  3. ${WHITE}./run_monitoring.sh start${NC} # Démarrer le monitoring"
    echo -e "  4. ${WHITE}./run_tests.sh all${NC}        # Tests complets"
    echo -e "  5. ${WHITE}./demo_presentation.sh${NC}    # Présentation"
    echo ""

    echo -e "${GREEN}📋 Cahier des charges - Conformité:${NC}"
    echo -e "  ✅ DVC - Gestion données"
    echo -e "  ✅ MLflow - Tracking"
    echo -e "  ✅ PyTorch Lightning"
    echo -e "  ✅ FastAPI - API REST"
    echo -e "  ✅ Docker - Conteneurisation"
    echo -e "  ✅ Kubernetes - Orchestration"
    echo -e "  ✅ GitHub Actions - CI/CD"
    echo -e "  ✅ Prometheus - Métriques"
    echo -e "  ✅ Grafana - Dashboard"
    echo -e "  ✅ Evidently - Drift Detection"
    echo -e "  ✅ Tests unitaires"
    echo -e "  ✅ Export ONNX"
    echo -e "  ✅ Temps réponse < 2s"
    echo ""

    echo -e "${BLUE}💡 Conseils:${NC}"
    echo -e "  • Lancez ${WHITE}./run_pipeline.sh check${NC} pour vérifier l'installation"
    echo -e "  • Utilisez ${WHITE}./run_api.sh test${NC} pour tester l'API rapidement"
    echo -e "  • Consultez ${WHITE}../docs/GUIDE_PRESENTATION.md${NC} pour la soutenance"
    echo ""
}

# Fonction principale
main() {
    # Vérifier que nous sommes dans le bon répertoire
    if [ ! -f "../dvc.yaml" ]; then
        echo -e "${RED}Erreur: Lancez ce script depuis le dossier scripts/${NC}"
        echo -e "${YELLOW}Usage: cd scripts && ./main.sh${NC}"
        exit 1
    fi

    # Afficher le menu
    show_menu

    # Si un argument est passé, l'exécuter
    if [ $# -gt 0 ]; then
        case "$1" in
            "pipeline")
                echo -e "${YELLOW}Lancement du pipeline complet...${NC}"
                ./run_pipeline.sh all
                ;;
            "api")
                echo -e "${YELLOW}Lancement de l'API...${NC}"
                ./run_api.sh dev
                ;;
            "monitoring")
                echo -e "${YELLOW}Lancement du monitoring...${NC}"
                ./run_monitoring.sh start
                ;;
            "tests")
                echo -e "${YELLOW}Lancement des tests...${NC}"
                ./run_tests.sh all
                ;;
            "demo")
                echo -e "${YELLOW}Lancement de la démonstration...${NC}"
                ./demo_presentation.sh
                ;;
            *)
                echo -e "${RED}Argument inconnu: $1${NC}"
                echo -e "${YELLOW}Utilisez sans argument pour voir le menu${NC}"
                exit 1
                ;;
        esac
    fi
}

# Lancer la fonction principale
main "$@"