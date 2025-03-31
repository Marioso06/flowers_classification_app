"""
Flask API for flower classification with explainable AI capabilities
"""
import os
import time
import logging
import traceback
import json
import base64
from datetime import datetime
from flask import Flask, jsonify, request, send_file

# Import from modular files
from src.utils.image_processing import process_base64_image
from src.utils.model_utils import predict, load_models

# Import explanation methods
from src.explanations.shap_explanations import generate_gradient_shap_explanation, generate_deep_shap_explanation
from src.explanations.lime_explanations import generate_lime_explanation
from src.explanations.fallback import generate_custom_shap_explanation, generate_fallback_explanation

# Import metrics
from prometheus_flask_exporter import PrometheusMetrics
from src.metrics.prometheus_metrics import (
    prediction_requests, prediction_time, explanation_requests, 
    explanation_time, explanation_failures, start_monitoring_thread
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Get project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Ensure static directory exists
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Load models
model_v1, model_v2 = load_models()

# Routes for serving static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (images, etc.)"""
    return send_file(os.path.join(STATIC_DIR, filename))

@app.route('/flowers_classification_home', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    app_info = {
        "name": "Flowers Classification API with Explainable AI",
        "description": "This API takes an image and returns the top K categories based on 102 Flowers Categories, with options for explainable AI visualizations",
        "version": "v1.0",
        "endpoints": {
            "/flowers_classification_home": "Home Page",
            "/health_status" : "Check APIs Health",
            "/v1/predict": "This version of the API is based on VGG19",
            "/v2/predict": "This version of the API is based on VGG13",
            "/v1/explain_shap": "Get SHAP explanations for model v1 predictions",
            "/v2/explain_shap": "Get SHAP explanations for model v2 predictions",
            "/v1/explain_lime": "Get LIME explanations for model v1 predictions",
            "/v2/explain_lime": "Get LIME explanations for model v2 predictions"
        },
        "input_format" : {
            "image_data": "Give the image to predict as a Base64 encoded string",
            "top_k": "Number of classes to return"
        },
        "example_request": {
            "image_data": "base64_enconded_image_string",
            "top_k": 5           
        },
        "example_response": {
            "sucess": True,
            "prediction":{
                "probabilities": [0.9,0.5,0.4,0.42,0.01],
                "classes": ["21","22","86","96","3"],
                "classes_name": ["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Flower 5",]
            }
        },
        "xai_explanation": {
            "SHAP": "SHapley Additive exPlanations - Provides pixel-level contributions to the prediction",
            "LIME": "Local Interpretable Model-agnostic Explanations - Identifies regions that influence the prediction",
            "visualization": "Each explanation endpoint returns a base64-encoded image visualization"
        }
    }
    
    return jsonify(app_info)

@app.route('/health_status', methods=['GET'])
def health_status():
    """API health status endpoint"""
    # Check if models are loaded
    models_status = {
        "model_v1": model_v1 is not None,
        "model_v2": model_v2 is not None
    }
    
    if all(models_status.values()):
        status = "healthy"
    else:
        status = "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models": models_status
    })

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict using model v1"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received prediction request for model v1")
    prediction_requests.labels(model_version="v1").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v1 is None:
        logger.error("Model v1 is not loaded")
        return jsonify({"error": "Model v1 is not available"}), 503
    
    try:
        # Process image
        np_image = process_base64_image(base64_image)
        
        # Make prediction
        prediction_result = predict(np_image, model_v1, top_k)
        
        # Log timing metric
        prediction_time.labels(model_version="v1").observe(time.time() - start_time)
        
        logger.info("Prediction successful")
        return jsonify({
            "success": True,
            "prediction": prediction_result
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict using model v2"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received prediction request for model v2")
    prediction_requests.labels(model_version="v2").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        # Process image
        np_image = process_base64_image(base64_image)
        
        # Make prediction
        prediction_result = predict(np_image, model_v2, top_k)
        
        # Log timing metric
        prediction_time.labels(model_version="v2").observe(time.time() - start_time)
        
        logger.info("Prediction successful")
        return jsonify({
            "success": True,
            "prediction": prediction_result
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v1/explain_shap', methods=['POST'])
def explain_shap_v1():
    """Generate SHAP explanation for model v1"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received SHAP explanation request for model v1")
    explanation_requests.labels(explanation_type="SHAP", model_version="v1").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v1 is None:
        logger.error("Model v1 is not loaded")
        return jsonify({"error": "Model v1 is not available"}), 503
    
    try:
        logger.info(f"Processing image for SHAP explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Try to use gradient SHAP first
        try:
            explanation = generate_gradient_shap_explanation(np_image, model_v1, top_k)
        except Exception as e:
            logger.warning(f"GradientSHAP failed, trying DeepSHAP: {str(e)}")
            try:
                explanation = generate_deep_shap_explanation(np_image, model_v1, top_k)
            except Exception as e:
                logger.warning(f"DeepSHAP failed, using custom SHAP implementation: {str(e)}")
                explanation = generate_custom_shap_explanation(np_image, model_v1, top_k)
        
        # Log timing metric
        explanation_time.labels(explanation_type="SHAP", model_version="v1").observe(time.time() - start_time)
        
        logger.info("SHAP explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        # Log the failure
        explanation_failures.labels(explanation_type="SHAP", model_version="v1").inc()
        logger.error(f"Error during SHAP explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/explain_shap', methods=['POST'])
def explain_shap_v2():
    """Generate SHAP explanation for model v2"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received SHAP explanation request for model v2")
    explanation_requests.labels(explanation_type="SHAP", model_version="v2").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        logger.info(f"Processing image for SHAP explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Try to use gradient SHAP first
        try:
            explanation = generate_gradient_shap_explanation(np_image, model_v2, top_k)
        except Exception as e:
            logger.warning(f"GradientSHAP failed, trying DeepSHAP: {str(e)}")
            try:
                explanation = generate_deep_shap_explanation(np_image, model_v2, top_k)
            except Exception as e:
                logger.warning(f"DeepSHAP failed, using custom SHAP implementation: {str(e)}")
                explanation = generate_custom_shap_explanation(np_image, model_v2, top_k)
        
        # Log timing metric
        explanation_time.labels(explanation_type="SHAP", model_version="v2").observe(time.time() - start_time)
        
        logger.info("SHAP explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        # Log the failure
        explanation_failures.labels(explanation_type="SHAP", model_version="v2").inc()
        logger.error(f"Error during SHAP explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v1/explain_lime', methods=['POST'])
def explain_lime_v1():
    """Generate LIME explanation for model v1"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received LIME explanation request for model v1")
    explanation_requests.labels(explanation_type="LIME", model_version="v1").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v1 is None:
        logger.error("Model v1 is not loaded")
        return jsonify({"error": "Model v1 is not available"}), 503
    
    try:
        logger.info(f"Processing image for LIME explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(np_image, model_v1, top_k)
        
        # Log timing metric
        explanation_time.labels(explanation_type="LIME", model_version="v1").observe(time.time() - start_time)
        
        logger.info("LIME explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        # Log the failure
        explanation_failures.labels(explanation_type="LIME", model_version="v1").inc()
        logger.error(f"Error during LIME explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/explain_lime', methods=['POST'])
def explain_lime_v2():
    """Generate LIME explanation for model v2"""
    # Start timer for metrics
    start_time = time.time()
    
    # Log the request
    logger.info("Received LIME explanation request for model v2")
    explanation_requests.labels(explanation_type="LIME", model_version="v2").inc()
    
    # Parse request data
    data = request.json
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "No JSON data provided"}), 400
    
    if 'image_data' not in data:
        logger.warning("Missing required field: image_data")
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    # Check if model is loaded
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        logger.info(f"Processing image for LIME explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(np_image, model_v2, top_k)
        
        # Log timing metric
        explanation_time.labels(explanation_type="LIME", model_version="v2").observe(time.time() - start_time)
        
        logger.info("LIME explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        # Log the failure
        explanation_failures.labels(explanation_type="LIME", model_version="v2").inc()
        logger.error(f"Error during LIME explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Start resource monitoring in a background thread
    monitor_thread = start_monitoring_thread()
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')  # Use 0.0.0.0 to listen on all interfaces in container
    port = int(os.environ.get('FLASK_PORT', 9000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flowers Classification API on {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)
