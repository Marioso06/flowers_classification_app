#!/usr/bin/env python3
"""
Stress Test for Flowers Classification API
------------------------------------------
This script performs stress testing on the flowers classification API by:
1. Sending test images to both v1 and v2 endpoints
2. Comparing predictions with actual flower classes
3. Measuring response times and accuracy
4. Reporting performance metrics and visualization
"""

import os
import requests
import base64
import json
import time
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import seaborn as sns
from PIL import Image
from io import BytesIO
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API endpoints
DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = 9000
API_BASE_URL = "http://{host}:{port}"
API_V1_ENDPOINT = "/v1/predict"
API_V2_ENDPOINT = "/v2/predict"

# Path to test images
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "tests", "archive", "test_images")

# Map folder names to flower indices
# This will be filled dynamically based on cat_to_name.json
FLOWER_NAME_TO_INDEX = {}


def load_category_mapping():
    """Load the category to name mapping from the JSON file"""
    cat_to_name_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "configs", "cat_to_name.json")
    try:
        with open(cat_to_name_path, 'r') as f:
            cat_to_name = json.load(f)
            
        # Create a reverse mapping from name to index
        name_to_idx = {}
        for idx, name in cat_to_name.items():
            name_to_idx[name.lower()] = idx
            
        return cat_to_name, name_to_idx
    except Exception as e:
        logger.error(f"Error loading category mapping: {e}")
        return {}, {}


def encode_image(image_path):
    """Encode the image at the specified path to base64"""
    try:
        # Open image with PIL first to ensure it's valid and readable
        img = Image.open(image_path)
        
        # Convert to RGB if needed (handles PNG with alpha channel)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # Save to BytesIO object to ensure it's a valid JPEG
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        
        # Get bytes and encode to base64
        img_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        logger.debug(f"Successfully encoded image {image_path} to base64 (length: {len(base64_encoded)})")
        return base64_encoded
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def predict_image(base64_image, endpoint, host, port, top_k=5):
    """Send an image to the prediction API and return the results"""
    url = f"{API_BASE_URL.format(host=host, port=port)}{endpoint}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    payload = {
        'image_data': base64_image,
        'top_k': top_k
    }
    
    # Debug info
    img_length = len(base64_image) if base64_image else 0
    logger.debug(f"Sending image with base64 length {img_length} to {endpoint}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': result.get('success', False),
                'prediction': result.get('prediction', {}),
                'response_time': elapsed_time,
                'status_code': response.status_code
            }
        else:
            # More detailed error logging
            err_msg = response.text if response.text else "No response text"
            logger.warning(f"API request failed with status {response.status_code}: {err_msg}")
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {err_msg}",
                'response_time': elapsed_time,
                'status_code': response.status_code
            }
    except requests.exceptions.Timeout:
        logger.warning(f"Request to {endpoint} timed out")
        return {
            'success': False,
            'error': 'Request timed out',
            'response_time': 30.0,  # timeout value
            'status_code': 0
        }
    except Exception as e:
        logger.error(f"Error sending request to {endpoint}: {e}")
        return {
            'success': False,
            'error': str(e),
            'response_time': time.time() - start_time,
            'status_code': 0
        }


def process_image_folder(folder_name, folder_path, endpoint, host, port, concurrent_requests, max_images=None, request_delay=0.0):
    """Process all images in a folder and return results"""
    image_files = [f for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        logger.warning(f"No image files found in {folder_path}")
        return []
    
    # Limit the number of images if max_images is specified
    if max_images is not None and max_images > 0 and len(image_files) > max_images:
        logger.info(f"Limiting to {max_images} images out of {len(image_files)} in folder {folder_name}")
        # Randomly sample to get a representative set
        image_files = np.random.choice(image_files, max_images, replace=False).tolist()
    
    results = []
    
    # Get the expected flower class (from folder name)
    expected_class = folder_name.lower()
    expected_index = FLOWER_NAME_TO_INDEX.get(expected_class, "unknown")
    
    logger.info(f"Processing {len(image_files)} images from '{folder_name}' (class index: {expected_index})")
    
    # Function to process a single image
    def process_single_image(image_file):
        image_path = os.path.join(folder_path, image_file)
        base64_image = encode_image(image_path)
        
        if base64_image:
            # Add delay between requests if specified
            if request_delay > 0:
                time.sleep(request_delay)
                
            # Get prediction from API
            api_result = predict_image(base64_image, endpoint, host, port)
            
            # Check if prediction is correct (if successful)
            is_correct = False
            predicted_class = "unknown"
            confidence = 0.0
            
            if api_result['success'] and 'prediction' in api_result:
                # Get the top prediction
                if 'classes' in api_result['prediction'] and api_result['prediction']['classes']:
                    predicted_class = api_result['prediction']['classes'][0]
                    if 'probabilities' in api_result['prediction'] and api_result['prediction']['probabilities']:
                        confidence = api_result['prediction']['probabilities'][0]
                    
                    # Check if the prediction matches the expected class
                    is_correct = (predicted_class == expected_index)
            
            # Return result data
            return {
                'image_path': image_path,
                'folder_name': folder_name,
                'expected_class': expected_class,
                'expected_index': expected_index,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct,
                'response_time': api_result.get('response_time', 0),
                'success': api_result.get('success', False),
                'status_code': api_result.get('status_code', 0),
                'error': api_result.get('error', '') if not api_result.get('success', False) else ''
            }
        else:
            return {
                'image_path': image_path,
                'folder_name': folder_name,
                'expected_class': expected_class,
                'expected_index': expected_index,
                'error': 'Failed to encode image',
                'success': False,
                'correct': False
            }
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # Submit all tasks and create a mapping
        future_to_image = {
            executor.submit(process_single_image, img): img 
            for img in image_files
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_image), 
                          total=len(image_files), 
                          desc=f"Processing {folder_name}"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
    
    return results


def run_stress_test(host, port, endpoint, concurrent_requests, samples_per_class=None, max_images=None, request_delay=0.0):
    """Run a stress test on the API using test images"""
    global FLOWER_NAME_TO_INDEX
    
    # Load category mapping
    cat_to_name, name_to_idx = load_category_mapping()
    FLOWER_NAME_TO_INDEX = name_to_idx
    
    # Check if test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        logger.error(f"Test images directory not found: {TEST_IMAGES_DIR}")
        return []
    
    # Get list of class folders
    class_folders = [f for f in os.listdir(TEST_IMAGES_DIR) 
                    if os.path.isdir(os.path.join(TEST_IMAGES_DIR, f))]
    
    if not class_folders:
        logger.error(f"No class folders found in {TEST_IMAGES_DIR}")
        return []
    
    all_results = []
    
    # Process each class folder
    for folder in class_folders:
        folder_path = os.path.join(TEST_IMAGES_DIR, folder)
        
        # Process the folder and get results
        results = process_image_folder(folder, folder_path, endpoint, host, port, 
                                      concurrent_requests, max_images, request_delay)
        
        # If sampling is requested, randomly select a subset of successful results
        if samples_per_class and len(results) > samples_per_class:
            # Filter for only successful results if possible
            successful_results = [r for r in results if r.get('success', False)]
            if len(successful_results) >= samples_per_class:
                results = np.random.choice(successful_results, samples_per_class, replace=False).tolist()
            else:
                # Not enough successful results, sample from all results
                results = np.random.choice(results, samples_per_class, replace=False).tolist()
        
        all_results.extend(results)
    
    return all_results


def analyze_results(results, output_dir='results'):
    """Analyze the test results and generate reports"""
    if not results:
        logger.error("No results to analyze")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Calculate overall statistics
    total_requests = len(df)
    successful_requests = df['success'].sum()
    failed_requests = total_requests - successful_requests
    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
    
    # Calculate accuracy (only for successful requests)
    correct_predictions = df[df['success']]['correct'].sum()
    total_valid_predictions = df['success'].sum()
    accuracy = (correct_predictions / total_valid_predictions) * 100 if total_valid_predictions > 0 else 0
    
    # Calculate response time statistics
    mean_response_time = df['response_time'].mean()
    median_response_time = df['response_time'].median()
    max_response_time = df['response_time'].max()
    min_response_time = df['response_time'].min()
    p95_response_time = df['response_time'].quantile(0.95)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("API STRESS TEST RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Requests: {total_requests}")
    logger.info(f"Successful Requests: {successful_requests} ({success_rate:.2f}%)")
    logger.info(f"Failed Requests: {failed_requests} ({100-success_rate:.2f}%)")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Mean Response Time: {mean_response_time:.3f} seconds")
    logger.info(f"Median Response Time: {median_response_time:.3f} seconds")
    logger.info(f"95th Percentile Response Time: {p95_response_time:.3f} seconds")
    logger.info(f"Min/Max Response Time: {min_response_time:.3f}/{max_response_time:.3f} seconds")
    logger.info("="*60)
    
    # Generate visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Response Time Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['response_time'], kde=True)
    plt.title('Response Time Distribution')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Frequency')
    
    # 2. Accuracy by Class
    plt.subplot(2, 2, 2)
    class_accuracy = df[df['success']].groupby('folder_name')['correct'].mean() * 100
    class_accuracy.plot(kind='bar')
    plt.title('Accuracy by Flower Class')
    plt.xlabel('Flower Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # 3. Success Rate by Class
    plt.subplot(2, 2, 3)
    success_by_class = df.groupby('folder_name')['success'].mean() * 100
    success_by_class.plot(kind='bar')
    plt.title('API Success Rate by Flower Class')
    plt.xlabel('Flower Class')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    
    # 4. Response Time by Class
    plt.subplot(2, 2, 4)
    sns.boxplot(x='folder_name', y='response_time', data=df)
    plt.title('Response Time by Flower Class')
    plt.xlabel('Flower Class')
    plt.ylabel('Response Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stress_test_results.png'))
    logger.info(f"Results visualization saved to {os.path.join(output_dir, 'stress_test_results.png')}")
    
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'stress_test_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")
    
    # Return summary metrics
    return {
        'total_requests': total_requests,
        'success_rate': success_rate,
        'accuracy': accuracy,
        'mean_response_time': mean_response_time,
        'p95_response_time': p95_response_time
    }


def main():
    """Main function to run the API stress test"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Stress test for Flowers Classification API')
    parser.add_argument('--host', default=DEFAULT_API_HOST, help=f'API host (default: {DEFAULT_API_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_API_PORT, help=f'API port (default: {DEFAULT_API_PORT})')
    parser.add_argument('--endpoint', choices=['v1', 'v2', 'both'], default='both', 
                        help='API endpoint version to test (default: both)')
    parser.add_argument('--concurrent', type=int, default=5, 
                        help='Number of concurrent requests (default: 5)')
    parser.add_argument('--samples', type=int, default=None, 
                        help='Number of samples per class (default: all)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to test per folder (default: all)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between API requests in seconds (default: 0.0)')
    parser.add_argument('--output', default='results', 
                        help='Output directory for results (default: results)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"Starting API Stress Test at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API Endpoint: http://{args.host}:{args.port}")
    logger.info(f"Test Images Directory: {TEST_IMAGES_DIR}")
    logger.info(f"Concurrent Requests: {args.concurrent}")
    logger.info(f"Samples per Class: {'All' if args.samples is None else args.samples}")
    logger.info(f"Max Images per Folder: {'All' if args.max_images is None else args.max_images}")
    logger.info(f"Request Delay: {args.delay} seconds")
    logger.info("="*80)
    
    # Create results directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run tests based on endpoint choice
    if args.endpoint in ['v1', 'both']:
        logger.info("\nTesting V1 API Endpoint...")
        v1_results = run_stress_test(
            host=args.host,
            port=args.port,
            endpoint=API_V1_ENDPOINT,
            concurrent_requests=args.concurrent,
            samples_per_class=args.samples,
            max_images=args.max_images,
            request_delay=args.delay
        )
        logger.info(f"V1 API Test completed with {len(v1_results)} requests")
        if v1_results:
            logger.info("Analyzing V1 API results...")
            v1_summary = analyze_results(v1_results, os.path.join(args.output, 'v1'))
    
    if args.endpoint in ['v2', 'both']:
        logger.info("\nTesting V2 API Endpoint...")
        v2_results = run_stress_test(
            host=args.host,
            port=args.port,
            endpoint=API_V2_ENDPOINT,
            concurrent_requests=args.concurrent,
            samples_per_class=args.samples,
            max_images=args.max_images,
            request_delay=args.delay
        )
        logger.info(f"V2 API Test completed with {len(v2_results)} requests")
        if v2_results:
            logger.info("Analyzing V2 API results...")
            v2_summary = analyze_results(v2_results, os.path.join(args.output, 'v2'))
    
    # Compare V1 and V2 if both were tested
    if args.endpoint == 'both' and 'v1_summary' in locals() and 'v2_summary' in locals():
        logger.info("\n" + "="*60)
        logger.info("COMPARISON: V1 vs V2 API ENDPOINTS")
        logger.info("="*60)
        logger.info(f"V1 Accuracy: {v1_summary['accuracy']:.2f}% | V2 Accuracy: {v2_summary['accuracy']:.2f}%")
        logger.info(f"V1 Success Rate: {v1_summary['success_rate']:.2f}% | V2 Success Rate: {v2_summary['success_rate']:.2f}%")
        logger.info(f"V1 Mean Response Time: {v1_summary['mean_response_time']:.3f}s | V2: {v2_summary['mean_response_time']:.3f}s")
        logger.info(f"V1 P95 Response Time: {v1_summary['p95_response_time']:.3f}s | V2: {v2_summary['p95_response_time']:.3f}s")
        
        # Create comparison chart
        plt.figure(figsize=(12, 6))
        
        # Compare accuracy and success rate
        plt.subplot(1, 2, 1)
        metrics = ['Accuracy', 'Success Rate']
        v1_values = [v1_summary['accuracy'], v1_summary['success_rate']]
        v2_values = [v2_summary['accuracy'], v2_summary['success_rate']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, v1_values, width, label='V1 API')
        plt.bar(x + width/2, v2_values, width, label='V2 API')
        
        plt.title('Accuracy and Success Rate Comparison')
        plt.ylabel('Percentage (%)')
        plt.xticks(x, metrics)
        plt.legend()
        
        # Compare response times
        plt.subplot(1, 2, 2)
        metrics = ['Mean', '95th Percentile']
        v1_values = [v1_summary['mean_response_time'], v1_summary['p95_response_time']]
        v2_values = [v2_summary['mean_response_time'], v2_summary['p95_response_time']]
        
        x = np.arange(len(metrics))
        
        plt.bar(x - width/2, v1_values, width, label='V1 API')
        plt.bar(x + width/2, v2_values, width, label='V2 API')
        
        plt.title('Response Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(x, metrics)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'api_comparison.png'))
        logger.info(f"Comparison chart saved to {os.path.join(args.output, 'api_comparison.png')}")
    
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"API Stress Test Completed in {total_time:.2f} seconds")
    logger.info("="*80)


if __name__ == "__main__":
    main()
