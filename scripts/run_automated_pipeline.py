#!/usr/bin/env python3
"""
Pipeline automatisé complet:
1. Détecte les changements DVC
2. Exécute le pipeline DVC (prepare_data -> train -> evaluate -> export)
3. Enregistre les données et modèles dans MLflow
4. Reconstruit l'image Docker avec le nouveau modèle
5. Redéploie localement avec Docker Compose
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutomatedPipeline:
    """Pipeline automatisé MLOps."""
    
    def __init__(self, config_path: str = "config.yaml", force: bool = False):
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent
        self.force = force
        
        # Charger la configuration
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Chemins importants
        self.dvc_yaml = self.project_root / "dvc.yaml"
        self.models_dir = self.project_root / "models"
        self.production_model = self.models_dir / "production" / "model.ckpt"
        self.docker_compose = self.project_root / "docker" / "docker-compose.yml"
        
    def check_dvc_changes(self) -> bool:
        """Vérifier si des changements DVC sont détectés."""
        logger.info("🔍 Vérification des changements DVC...")
        
        try:
            # Import relatif depuis le même répertoire
            sys.path.insert(0, str(Path(__file__).parent))
            from monitor_dvc_changes import DVCChangeMonitor
            monitor = DVCChangeMonitor(project_root=str(self.project_root))
            has_changes = monitor.has_changes()
            
            if has_changes:
                logger.info("✅ Changements DVC détectés!")
            else:
                logger.info("ℹ️  Aucun changement DVC détecté")
            
            return has_changes or self.force
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors de la vérification DVC: {e}")
            logger.info("ℹ️  Continuation avec l'option --force")
            return self.force
    
    def run_dvc_pipeline(self) -> bool:
        """Exécuter le pipeline DVC complet."""
        logger.info("🚀 Exécution du pipeline DVC...")
        
        if not self.dvc_yaml.exists():
            logger.error(f"❌ dvc.yaml non trouvé: {self.dvc_yaml}")
            return False
        
        try:
            # Exécuter dvc repro pour exécuter tout le pipeline
            logger.info("📊 Exécution: dvc repro")
            result = subprocess.run(
                ["dvc", "repro"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("✅ Pipeline DVC exécuté avec succès")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Erreur lors de l'exécution du pipeline DVC:")
                logger.error(result.stderr)
                return False
                
        except FileNotFoundError:
            logger.error("❌ DVC n'est pas installé. Installez-le avec: pip install dvc")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution du pipeline DVC: {e}")
            return False
    
    def verify_mlflow_registration(self) -> bool:
        """Vérifier que le modèle est enregistré dans MLflow."""
        logger.info("📝 Vérification de l'enregistrement MLflow...")
        
        mlflow_config = self.config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
        
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            
            # Vérifier la connexion
            experiments = mlflow.search_experiments()
            logger.info(f"✅ MLflow connecté: {len(experiments)} expériences trouvées")
            
            # Vérifier le dernier run
            experiment_name = mlflow_config.get("experiment_name", "plant_disease_detection")
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        max_results=1
                    )
                    if not runs.empty:
                        latest_run = runs.iloc[0]
                        logger.info(f"✅ Dernier run MLflow: {latest_run['run_id']}")
                        logger.info(f"   Métriques: {latest_run.get('metrics.val_acc', 'N/A')}")
                        return True
            except Exception as e:
                logger.warning(f"⚠️  Impossible de vérifier les runs: {e}")
            
            return True
        except ImportError:
            logger.warning("⚠️  MLflow non disponible")
            return True  # Ne pas bloquer si MLflow n'est pas disponible
        except Exception as e:
            logger.warning(f"⚠️  Erreur MLflow: {e}")
            return True  # Ne pas bloquer
    
    def verify_model_production(self) -> bool:
        """Vérifier que le modèle de production existe."""
        logger.info("🔍 Vérification du modèle de production...")
        
        if self.production_model.exists():
            size_mb = self.production_model.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Modèle de production trouvé: {self.production_model}")
            logger.info(f"   Taille: {size_mb:.2f} MB")
            return True
        else:
            logger.error(f"❌ Modèle de production non trouvé: {self.production_model}")
            logger.info("💡 Le modèle devrait être créé lors de l'entraînement")
            return False
    
    def build_docker_image(self) -> bool:
        """Reconstruire l'image Docker avec le nouveau modèle."""
        logger.info("🐳 Reconstruction de l'image Docker...")
        
        dockerfile = self.project_root / "docker" / "Dockerfile.inference"
        if not dockerfile.exists():
            logger.error(f"❌ Dockerfile non trouvé: {dockerfile}")
            return False
        
        image_name = self.config.get("docker", {}).get("image_name", "plant-disease-mlops")
        image_tag = self.config.get("docker", {}).get("tag", "latest")
        full_image_name = f"{image_name}:{image_tag}"
        
        try:
            # Vérifier que le modèle existe avant de construire
            if not self.verify_model_production():
                logger.error("❌ Impossible de construire l'image: modèle manquant")
                return False
            
            logger.info(f"🔨 Construction de l'image: {full_image_name}")
            
            result = subprocess.run(
                [
                    "docker", "build",
                    "-f", str(dockerfile),
                    "-t", full_image_name,
                    str(self.project_root)
                ],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("✅ Image Docker construite avec succès")
                return True
            else:
                logger.error(f"❌ Erreur lors de la construction de l'image:")
                logger.error(result.stderr)
                return False
                
        except FileNotFoundError:
            logger.error("❌ Docker n'est pas installé ou non disponible")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction Docker: {e}")
            return False
    
    def deploy_locally(self) -> bool:
        """Redéployer localement avec Docker Compose."""
        logger.info("🚀 Déploiement local avec Docker Compose...")
        
        if not self.docker_compose.exists():
            logger.error(f"❌ docker-compose.yml non trouvé: {self.docker_compose}")
            return False
        
        try:
            # Arrêter les services existants
            logger.info("🛑 Arrêt des services existants...")
            subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose), "down"],
                cwd=self.project_root,
                capture_output=True,
                check=False
            )
            
            # Démarrer les services
            logger.info("▶️  Démarrage des services...")
            result = subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose), "up", "-d"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("✅ Services déployés avec succès")
                logger.info("📊 Vérification du statut...")
                
                # Attendre un peu pour que les services démarrent
                time.sleep(5)
                
                # Vérifier le statut
                status_result = subprocess.run(
                    ["docker-compose", "-f", str(self.docker_compose), "ps"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=False
                )
                logger.info(status_result.stdout)
                
                return True
            else:
                logger.error(f"❌ Erreur lors du déploiement:")
                logger.error(result.stderr)
                return False
                
        except FileNotFoundError:
            logger.error("❌ docker-compose n'est pas installé")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors du déploiement: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Exécuter le pipeline complet."""
        logger.info("=" * 60)
        logger.info("🚀 DÉMARRAGE DU PIPELINE AUTOMATISÉ MLOPS")
        logger.info("=" * 60)
        
        steps = [
            ("Vérification DVC", self.check_dvc_changes),
            ("Pipeline DVC", self.run_dvc_pipeline),
            ("Vérification MLflow", self.verify_mlflow_registration),
            ("Vérification Modèle", self.verify_model_production),
            ("Construction Docker", self.build_docker_image),
            ("Déploiement Local", self.deploy_locally),
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            logger.info("")
            logger.info(f"📌 Étape: {step_name}")
            logger.info("-" * 60)
            
            try:
                start_time = time.time()
                success = step_func()
                elapsed = time.time() - start_time
                
                results[step_name] = {
                    "success": success,
                    "elapsed": elapsed
                }
                
                if not success:
                    logger.error(f"❌ Échec de l'étape: {step_name}")
                    logger.error("🛑 Arrêt du pipeline")
                    break
                else:
                    logger.info(f"✅ Étape terminée en {elapsed:.2f}s")
                    
            except Exception as e:
                logger.error(f"❌ Erreur dans l'étape {step_name}: {e}")
                results[step_name] = {
                    "success": False,
                    "error": str(e)
                }
                break
        
        # Résumé
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 RÉSUMÉ DU PIPELINE")
        logger.info("=" * 60)
        
        for step_name, result in results.items():
            status = "✅" if result.get("success") else "❌"
            elapsed = result.get("elapsed", 0)
            logger.info(f"{status} {step_name}: {elapsed:.2f}s")
        
        all_success = all(r.get("success", False) for r in results.values())
        
        if all_success:
            logger.info("")
            logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
            logger.info("")
            logger.info("📝 Services disponibles:")
            logger.info("   - API: http://localhost:8000")
            logger.info("   - MLflow: http://localhost:5000")
            logger.info("   - Grafana: http://localhost:3000")
            logger.info("   - Prometheus: http://localhost:9091")
        else:
            logger.error("")
            logger.error("❌ PIPELINE ÉCHOUÉ")
        
        return all_success


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Pipeline automatisé MLOps complet"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer l'exécution même sans changements DVC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--skip-dvc",
        action="store_true",
        help="Ignorer l'exécution du pipeline DVC"
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Ignorer la construction Docker"
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Ignorer le déploiement"
    )
    
    args = parser.parse_args()
    
    pipeline = AutomatedPipeline(config_path=args.config, force=args.force)
    
    # Modifier les méthodes si nécessaire
    if args.skip_dvc:
        pipeline.run_dvc_pipeline = lambda: True
    if args.skip_docker:
        pipeline.build_docker_image = lambda: True
    if args.skip_deploy:
        pipeline.deploy_locally = lambda: True
    
    success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

