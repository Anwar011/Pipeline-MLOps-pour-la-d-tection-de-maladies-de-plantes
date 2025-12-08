#!/bin/bash

# Manual deployment script for Azure AKS
# Usage: ./deploy-manual.sh

echo "🚀 Starting manual deployment to Azure AKS..."

# Set variables
IMAGE_TAG="manual-$(date +%Y%m%d-%H%M%S)"
ACR_NAME="plantdiseaseacr1674"
ACR_SERVER="plantdiseaseacr1674.azurecr.io"

# Step 1: Verify Azure login
echo "📋 Verifying Azure authentication..."
if ! az account show > /dev/null 2>&1; then
    echo "❌ Not logged into Azure. Please run: az login"
    exit 1
fi

echo "✅ Azure login verified: $(az account show --query user.name -o tsv)"

# Step 2: Login to ACR with fallback
echo "🔐 Logging into Azure Container Registry..."
if ! az acr login --name $ACR_NAME; then
    echo "⚠️ Direct ACR login failed, trying token method..."
    TOKEN=$(az acr login --name $ACR_NAME --expose-token --output tsv --query accessToken 2>/dev/null)
    if [ ! -z "$TOKEN" ]; then
        echo $TOKEN | docker login $ACR_SERVER -u 00000000-0000-0000-0000-000000000000 --password-stdin
        echo "✅ ACR login successful with token"
    else
        echo "❌ ACR login failed. Check permissions."
        exit 1
    fi
else
    echo "✅ ACR login successful"
fi

# Step 3: Build Docker image
echo "🏗️ Building Docker image..."
docker build -f docker/Dockerfile.inference \
    -t $ACR_SERVER/plant-disease-api:$IMAGE_TAG \
    -t $ACR_SERVER/plant-disease-api:latest \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Image built successfully"

# Step 4: Push to ACR
echo "📤 Pushing image to ACR..."
docker push $ACR_SERVER/plant-disease-api:$IMAGE_TAG
docker push $ACR_SERVER/plant-disease-api:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

echo "✅ Image pushed successfully"

# Step 5: Connect to AKS
echo "☸️ Connecting to AKS cluster..."
az aks get-credentials --resource-group plant-disease-rg --name plant-disease-aks --overwrite-existing

if [ $? -ne 0 ]; then
    echo "❌ AKS connection failed"
    exit 1
fi

echo "✅ Connected to AKS cluster"

# Step 6: Deploy to Kubernetes
echo "🚢 Deploying to Kubernetes..."

# Update image in deployment
kubectl set image deployment/plant-disease-api \
    api=$ACR_SERVER/plant-disease-api:$IMAGE_TAG \
    -n mlops

if [ $? -ne 0 ]; then
    echo "❌ Deployment update failed"
    exit 1
fi

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/plant-disease-api -n mlops --timeout=300s

# Check deployment status
echo "📊 Deployment status:"
kubectl get pods -n mlops
kubectl get svc -n mlops

echo "🎉 Deployment completed!"
echo "📍 Your API should be available at: http://$(kubectl get svc plant-disease-api-service -n mlops -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"