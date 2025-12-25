#!/usr/bin/env python3
"""
Script d'analyse de drift des données avec Evidently AI.
Conforme au cahier des charges: Surveillance des dérives de données.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_drift(config_path: str = "config.yaml"):
    """
    Analyse la dérive des données entre l'entraînement et la production.
    
    Cette fonction utilise Evidently AI pour:
    1. Détecter le drift de données (Data Drift)
    2. Détecter le drift de cibles (Target Drift)
    3. Générer un rapport HTML
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.error("❌ Evidently non installé. Installez avec: pip install evidently")
        create_demo_drift_report()
        return
    
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Créer le dossier de rapports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    logger.info("📊 Analyse de drift des données...")
    
    # Charger les données
    processed_path = Path(config["data"]["processed_path"])
    train_path = processed_path / "train"
    test_path = processed_path / "test"
    
    if not train_path.exists() or not test_path.exists():
        logger.warning("⚠️  Données non trouvées, création d'un rapport de démonstration")
        create_demo_drift_report()
        return
    
    # Extraire les features des images
    logger.info("🔍 Extraction des features des images...")
    train_features = extract_image_features(train_path)
    test_features = extract_image_features(test_path)
    
    if train_features is None or test_features is None:
        logger.warning("⚠️  Impossible d'extraire les features")
        create_demo_drift_report()
        return
    
    # Créer les DataFrames pour Evidently
    reference_df = pd.DataFrame(train_features)
    current_df = pd.DataFrame(test_features)
    
    # Configurer le column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "label"
    
    # Créer le rapport de drift
    logger.info("📈 Génération du rapport de drift...")
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    # Sauvegarder le rapport HTML
    report_path = reports_dir / "drift_report.html"
    report.save_html(str(report_path))
    
    # Extraire les métriques de drift
    drift_metrics = extract_drift_metrics(report)
    
    # Sauvegarder les métriques JSON
    metrics_path = reports_dir / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(drift_metrics, f, indent=2)
    
    logger.info(f"✅ Rapport de drift généré: {report_path}")
    logger.info(f"📊 Data Drift détecté: {drift_metrics.get('data_drift_detected', 'N/A')}")
    
    return drift_metrics


def extract_image_features(data_path: Path, max_samples: int = 500):
    """
    Extrait des features statistiques des images.
    
    Features extraites:
    - Moyenne RGB
    - Écart-type RGB
    - Luminosité moyenne
    - Contraste
    """
    features = []
    
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    for class_idx, class_dir in enumerate(sorted(class_dirs)):
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        # Limiter le nombre d'échantillons par classe
        samples_per_class = max_samples // len(class_dirs)
        image_files = image_files[:samples_per_class]
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                
                # Calculer les features
                feature = {
                    "mean_r": float(np.mean(img_array[:, :, 0])),
                    "mean_g": float(np.mean(img_array[:, :, 1])),
                    "mean_b": float(np.mean(img_array[:, :, 2])),
                    "std_r": float(np.std(img_array[:, :, 0])),
                    "std_g": float(np.std(img_array[:, :, 1])),
                    "std_b": float(np.std(img_array[:, :, 2])),
                    "brightness": float(np.mean(img_array)),
                    "contrast": float(np.std(img_array)),
                    "label": class_name
                }
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"⚠️  Erreur lors du traitement de {img_path}: {e}")
    
    if not features:
        return None
    
    return features


def extract_drift_metrics(report):
    """Extrait les métriques de drift du rapport Evidently."""
    try:
        result = report.as_dict()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "data_drift_detected": False,
            "drift_share": 0.0,
            "features_drifted": []
        }
        
        # Parser les résultats
        for metric in result.get("metrics", []):
            metric_result = metric.get("result", {})
            
            if "drift_share" in metric_result:
                metrics["drift_share"] = metric_result["drift_share"]
                metrics["data_drift_detected"] = metric_result.get("dataset_drift", False)
            
            if "drift_by_columns" in metric_result:
                for col, info in metric_result["drift_by_columns"].items():
                    if info.get("drift_detected", False):
                        metrics["features_drifted"].append(col)
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction des métriques: {e}")
        return {"error": str(e)}


def create_demo_drift_report():
    """Crée un rapport de démonstration."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Rapport HTML simple
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drift Analysis Report - Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #4CAF50; color: white; padding: 20px; }
            .section { margin: 20px 0; padding: 15px; background: #f9f9f9; }
            .metric { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; }
            .ok { color: green; }
            .warning { color: orange; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌱 Plant Disease Detection - Drift Analysis Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
        
        <div class="section">
            <h2>📊 Data Drift Summary</h2>
            <div class="metric">
                <h3>Data Drift Status</h3>
                <p class="ok">✅ No significant drift detected</p>
            </div>
            <div class="metric">
                <h3>Drift Share</h3>
                <p>0.15 (15% of features)</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Feature Analysis</h2>
            <table border="1" cellpadding="10">
                <tr><th>Feature</th><th>Drift Score</th><th>Status</th></tr>
                <tr><td>mean_r</td><td>0.12</td><td class="ok">OK</td></tr>
                <tr><td>mean_g</td><td>0.08</td><td class="ok">OK</td></tr>
                <tr><td>mean_b</td><td>0.15</td><td class="ok">OK</td></tr>
                <tr><td>brightness</td><td>0.10</td><td class="ok">OK</td></tr>
                <tr><td>contrast</td><td>0.22</td><td class="warning">Monitor</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            <ul>
                <li>Continue monitoring contrast feature</li>
                <li>No immediate retraining required</li>
                <li>Schedule next drift analysis in 7 days</li>
            </ul>
        </div>
        
        <p><em>Note: This is a demo report. Run with actual data for real analysis.</em></p>
    </body>
    </html>
    """
    
    with open(reports_dir / "drift_report.html", "w") as f:
        f.write(html_content)
    
    # Métriques JSON
    demo_metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_drift_detected": False,
        "drift_share": 0.15,
        "features_drifted": [],
        "note": "Demo metrics - run with actual data for real analysis"
    }
    
    with open(reports_dir / "drift_metrics.json", "w") as f:
        json.dump(demo_metrics, f, indent=2)
    
    logger.info("✅ Rapport de démonstration créé: reports/drift_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyser le drift des données")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    analyze_drift(args.config)
