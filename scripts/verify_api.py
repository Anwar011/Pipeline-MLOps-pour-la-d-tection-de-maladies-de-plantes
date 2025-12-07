#!/usr/bin/env python3
"""
Script to test the API with a sample image.
"""

import requests
from PIL import Image
import numpy as np
import io
import json

def create_test_image():
    """Create a simple test image (green leaf-like)."""
    # Create a 224x224 green image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = 100  # Green channel
    
    img = Image.fromarray(img_array)
    
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    return buf

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Plant Disease Detection API\n")
    
    # Test 1: Health check
    print("1️⃣ Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 2: Get classes
    print("2️⃣ Testing classes endpoint...")
    response = requests.get(f"{base_url}/classes")
    data = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Number of classes: {data['num_classes']}")
    print(f"   First 3 classes: {data['classes'][:3]}\n")
    
    # Test 3: Make prediction
    print("3️⃣ Testing prediction endpoint...")
    test_image = create_test_image()
    files = {'file': ('test.jpg', test_image, 'image/jpeg')}
    
    response = requests.post(f"{base_url}/predict", files=files)
    result = response.json()
    
    print(f"   Status: {response.status_code}")
    print(f"   Predicted disease: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Inference time: {result.get('inference_time_ms', 'N/A')} ms")
    
    # Show top 3 predictions
    print(f"\n   Top 3 predictions:")
    top5 = result['top5_predictions']
    # Take top 3
    for i, pred in enumerate(top5[:3], 1):
        print(f"      {i}. {pred['class']}: {pred['confidence']:.2%}")
    
    print("\n✅ All tests passed!")
    print(f"\n📖 API Documentation: {base_url}/docs")
    print(f"📊 Metrics: {base_url}/metrics")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is it running?")
        print("   Start with: python scripts/run_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")
