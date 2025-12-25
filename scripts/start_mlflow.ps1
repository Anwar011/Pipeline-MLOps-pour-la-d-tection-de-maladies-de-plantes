# Script to start/stop MLflow and monitoring services
# Usage: .\start_mlflow.ps1 [start|stop|status]

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "logs")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"

# Change to docker directory
$DockerDir = Join-Path $PSScriptRoot ".." "docker"
Push-Location $DockerDir

try {
    switch ($Action) {
        "start" {
            Write-Host "🚀 Starting MLflow and monitoring services..." -ForegroundColor Green
            
            # Start services
            docker-compose up -d mlflow-server prometheus grafana
            
            Write-Host ""
            Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            
            # Check status
            docker-compose ps
            
            Write-Host ""
            Write-Host "✅ Services started successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "📊 Access services at:" -ForegroundColor Cyan
            Write-Host "   - MLflow:     http://localhost:5000"
            Write-Host "   - Grafana:    http://localhost:3000 (admin/admin)"
            Write-Host "   - Prometheus: http://localhost:9091"
        }
        
        "stop" {
            Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
            docker-compose down
            Write-Host "✅ Services stopped" -ForegroundColor Green
        }
        
        "status" {
            Write-Host "📊 Service status:" -ForegroundColor Cyan
            docker-compose ps
        }
        
        "logs" {
            Write-Host "📋 Showing logs (Ctrl+C to exit)..." -ForegroundColor Cyan
            docker-compose logs -f mlflow-server
        }
    }
}
finally {
    Pop-Location
}
