#!/usr/bin/env python3
"""
Simple script to check the flowers classification API directly
"""
import requests
import base64
import json
import sys
from PIL import Image
from io import BytesIO
import argparse

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        # Open and process the image
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # Save as JPEG to BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        
        # Get base64 encoded string
        img_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        print(f"Successfully encoded image (length: {len(base64_encoded)})")
        return base64_encoded
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def check_api(image_path, endpoint, host="localhost", port=9000):
    """Check the API with a single image"""
    base64_image = encode_image(image_path)
    if not base64_image:
        print("Failed to encode image")
        return

    url = f"http://{host}:{port}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'image_data': base64_image,
        'top_k': 3
    }

    print(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check flowers classification API')
    parser.add_argument('--image', required=True, help='Path to an image file')
    parser.add_argument('--endpoint', default='/v1/predict', help='API endpoint (/v1/predict or /v2/predict)')
    parser.add_argument('--host', default='localhost', help='API host')
    parser.add_argument('--port', type=int, default=9000, help='API port')
    
    args = parser.parse_args()
    check_api(args.image, args.endpoint, args.host, args.port)

if __name__ == "__main__":
    main()
