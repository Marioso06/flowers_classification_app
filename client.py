#!/usr/bin/env python3
"""
Flowers Classification API Client

This script provides a simple command-line client for interacting with 
the Flowers Classification API. It allows users to send flower images for 
classification and displays the results.

Usage:
  python client.py --image path/to/image.jpg [--top_k 5] [--version v1]
"""

import argparse
import base64
import requests
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Flowers Classification API Client')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the flower image file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top classes to return (default: 5)')
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'],
                        help='API version to use (v1 or v2, default: v1)')
    parser.add_argument('--url', type=str, default='http://localhost:9000',
                        help='API base URL (default: http://localhost:9000)')
    parser.add_argument('--save', action='store_true',
                        help='Save the output visualization as an image')
    return parser.parse_args()

def encode_image(image_path):
    """Encode an image file as base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string

def predict_flower(image_path, top_k=5, version='v1', base_url='http://localhost:9000'):
    """Send a prediction request to the Flowers Classification API."""
    # Prepare API endpoint
    endpoint = f"{base_url}/{version}/predict"
    
    # Encode image
    try:
        encoded_image = encode_image(image_path)
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
    
    # Prepare request data
    data = {
        "image_data": encoded_image,
        "top_k": top_k
    }
    
    # Send request
    try:
        print(f"Sending request to {endpoint}...")
        response = requests.post(endpoint, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def display_results(result, image_path, save=False):
    """Display the prediction results and the image."""
    if not result or not result.get('success', False):
        print("No valid prediction results to display")
        if result and 'error' in result:
            print(f"Error: {result['error']}")
        return
    
    prediction = result['prediction']
    probabilities = prediction['probabilities']
    class_names = prediction['class_names']
    
    # Display image and prediction results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    image = Image.open(image_path)
    ax1.imshow(np.array(image))
    ax1.set_title(f"Predicted: {class_names[0]}")
    ax1.axis('off')
    
    # Display the prediction results
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    
    if save:
        image_name = os.path.basename(image_path)
        save_path = f"client_prediction_{image_name}.png"
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"Classifying image: {args.image}")
    print(f"Using API version: {args.version}")
    print(f"Requesting top {args.top_k} predictions")
    
    result = predict_flower(
        args.image, 
        top_k=args.top_k, 
        version=args.version,
        base_url=args.url
    )
    
    if result:
        print("\nPrediction Results:")
        if result.get('success', False):
            prediction = result['prediction']
            for i, (name, prob) in enumerate(zip(prediction['class_names'], prediction['probabilities'])):
                print(f"{i+1}. {name}: {prob:.4f}")
            
            # Display visualization
            display_results(result, args.image, save=args.save)
        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")
    else:
        print("Failed to get a response from the API")

if __name__ == "__main__":
    main()
