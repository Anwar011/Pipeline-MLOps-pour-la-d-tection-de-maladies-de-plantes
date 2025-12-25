#!/usr/bin/env python3
"""
Script de surveillance continue qui:
1. Surveille les changements DVC
2. Déclenche automatiquement le pipeline complet
3. Redéploie l'API avec le nouveau modèle
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineWatcher:
    """Surveillant qui déclenche le pipeline automatiquement."""
    
    def __init__(self, interval: int = 30, project_root: str = "."):
        self.interval = interval
        self.project_root = Path(project_root)
        self.running = True
        self.last_trigger_time = None
        
        # Gérer les signaux pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gérer les signaux d'arrêt."""
        logger.info("\n🛑 Arrêt demandé...")
        self.running = False
    
    def check_and_trigger(self) -> bool:
        """Vérifier les changements et déclencher le pipeline si nécessaire."""
        try:
            # Imports depuis le même répertoire
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from monitor_dvc_changes import DVCChangeMonitor
            
            monitor = DVCChangeMonitor(project_root=str(self.project_root))
            
            if monitor.has_changes():
                logger.info("")
                logger.info("=" * 60)
                logger.info("🔄 CHANGEMENTS DÉTECTÉS - DÉCLENCHEMENT DU PIPELINE")
                logger.info("=" * 60)
                
                # Importer et exécuter le pipeline
                from run_automated_pipeline import AutomatedPipeline
                
                pipeline = AutomatedPipeline(force=True)
                success = pipeline.run_full_pipeline()
                
                if success:
                    self.last_trigger_time = time.time()
                    logger.info("✅ Pipeline exécuté avec succès")
                else:
                    logger.error("❌ Pipeline échoué")
                
                return success
            else:
                logger.debug("Aucun changement détecté")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification: {e}")
            return False
    
    def run(self):
        """Exécuter la surveillance continue."""
        logger.info("=" * 60)
        logger.info("👀 SURVEILLANCE DVC AUTOMATIQUE ACTIVÉE")
        logger.info("=" * 60)
        logger.info(f"📊 Vérification toutes les {self.interval} secondes")
        logger.info("💡 Appuyez sur Ctrl+C pour arrêter")
        logger.info("")
        
        check_count = 0
        
        while self.running:
            try:
                check_count += 1
                logger.info(f"🔍 Vérification #{check_count}...")
                
                self.check_and_trigger()
                
                if self.running:
                    logger.info(f"⏳ Prochaine vérification dans {self.interval}s...")
                    logger.info("")
                    time.sleep(self.interval)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de surveillance: {e}")
                if self.running:
                    time.sleep(self.interval)
        
        logger.info("")
        logger.info("👋 Surveillance arrêtée")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Surveillance continue et déclenchement automatique du pipeline"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Intervalle de vérification en secondes (défaut: 30)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Racine du projet"
    )
    
    args = parser.parse_args()
    
    watcher = PipelineWatcher(interval=args.interval, project_root=args.project_root)
    watcher.run()


if __name__ == "__main__":
    main()

