# Script PowerShell de démarrage rapide pour le pipeline automatisé

Write-Host "🚀 Démarrage du pipeline automatisé MLOps" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier les prérequis
Write-Host "📋 Vérification des prérequis..." -ForegroundColor Yellow

# Vérifier Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python trouvé: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3 n'est pas installé" -ForegroundColor Red
    exit 1
}

# Vérifier DVC
try {
    $dvcVersion = dvc --version 2>&1
    Write-Host "✅ DVC trouvé: $dvcVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  DVC n'est pas installé. Installation..." -ForegroundColor Yellow
    pip install dvc
}

# Vérifier Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker trouvé: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker n'est pas installé" -ForegroundColor Red
    exit 1
}

# Vérifier Docker Compose
try {
    $composeVersion = docker-compose --version 2>&1
    Write-Host "✅ Docker Compose trouvé: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose n'est pas installé" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎯 Choisissez une option:" -ForegroundColor Cyan
Write-Host "1. Exécuter le pipeline une fois"
Write-Host "2. Démarrer la surveillance continue"
Write-Host "3. Vérifier les changements DVC uniquement"
Write-Host ""

$choice = Read-Host "Votre choix (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 Exécution du pipeline..." -ForegroundColor Green
        python scripts/run_automated_pipeline.py
    }
    "2" {
        Write-Host ""
        Write-Host "👀 Démarrage de la surveillance continue..." -ForegroundColor Green
        Write-Host "💡 Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Yellow
        python scripts/watch_and_trigger.py
    }
    "3" {
        Write-Host ""
        Write-Host "🔍 Vérification des changements DVC..." -ForegroundColor Green
        python scripts/monitor_dvc_changes.py
    }
    default {
        Write-Host "❌ Choix invalide" -ForegroundColor Red
        exit 1
    }
}

