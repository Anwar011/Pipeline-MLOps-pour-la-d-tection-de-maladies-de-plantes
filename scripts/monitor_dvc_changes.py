#!/usr/bin/env python3
"""
Script pour surveiller les changements de données DVC et déclencher le pipeline.
Utilise le système de fichiers pour détecter les modifications dans dvc.lock.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DVCChangeMonitor:
    """Moniteur de changements DVC."""
    
    def __init__(self, project_root: str = ".", state_file: str = ".dvc_monitor_state.json"):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / state_file
        self.dvc_lock_file = self.project_root / "dvc.lock"
        self.dvc_yaml_file = self.project_root / "dvc.yaml"
        self.state = self.load_state()
    
    def load_state(self) -> dict:
        """Charger l'état précédent."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'état: {e}")
        return {
            "dvc_lock_hash": None,
            "dvc_yaml_hash": None,
            "last_check": None
        }
    
    def save_state(self):
        """Sauvegarder l'état actuel."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {e}")
    
    def get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculer le hash d'un fichier."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Erreur lors du calcul du hash de {file_path}: {e}")
            return None
    
    def check_dvc_changes(self) -> bool:
        """Vérifier si DVC a détecté des changements."""
        current_lock_hash = self.get_file_hash(self.dvc_lock_file)
        current_yaml_hash = self.get_file_hash(self.dvc_yaml_file)
        
        # Comparer avec l'état précédent
        lock_changed = current_lock_hash != self.state.get("dvc_lock_hash")
        yaml_changed = current_yaml_hash != self.state.get("dvc_yaml_hash")
        
        if lock_changed or yaml_changed:
            logger.info("🔍 Changements DVC détectés!")
            if lock_changed:
                logger.info("  - dvc.lock modifié (nouvelles données)")
            if yaml_changed:
                logger.info("  - dvc.yaml modifié (pipeline modifié)")
            
            # Mettre à jour l'état
            self.state["dvc_lock_hash"] = current_lock_hash
            self.state["dvc_yaml_hash"] = current_yaml_hash
            self.state["last_check"] = time.time()
            self.save_state()
            
            return True
        
        return False
    
    def check_data_files_changed(self) -> bool:
        """Vérifier si des fichiers .dvc ont été modifiés."""
        dvc_files = list(self.project_root.glob("data/**/*.dvc"))
        dvc_files.extend(list(self.project_root.glob("*.dvc")))
        
        current_hashes = {}
        for dvc_file in dvc_files:
            file_hash = self.get_file_hash(dvc_file)
            if file_hash:
                current_hashes[str(dvc_file.relative_to(self.project_root))] = file_hash
        
        stored_hashes = self.state.get("dvc_file_hashes", {})
        
        if current_hashes != stored_hashes:
            logger.info("🔍 Fichiers .dvc modifiés détectés!")
            self.state["dvc_file_hashes"] = current_hashes
            self.save_state()
            return True
        
        return False
    
    def has_changes(self) -> bool:
        """Vérifier s'il y a des changements."""
        return self.check_dvc_changes() or self.check_data_files_changed()


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Surveiller les changements DVC et déclencher le pipeline"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Mode surveillance continue (polling)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Intervalle de vérification en secondes (mode watch)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Racine du projet"
    )
    
    args = parser.parse_args()
    
    monitor = DVCChangeMonitor(project_root=args.project_root)
    
    if args.watch:
        logger.info(f"👀 Surveillance DVC activée (vérification toutes les {args.interval}s)")
        logger.info("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                if monitor.has_changes():
                    logger.info("✅ Changements détectés! Le pipeline devrait être déclenché.")
                    logger.info("💡 Exécutez: python scripts/run_automated_pipeline.py")
                else:
                    logger.debug("Aucun changement détecté")
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("\n👋 Arrêt de la surveillance")
    else:
        # Vérification unique
        if monitor.has_changes():
            logger.info("✅ Changements DVC détectés!")
            sys.exit(0)
        else:
            logger.info("ℹ️  Aucun changement détecté")
            sys.exit(1)


if __name__ == "__main__":
    main()

