import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import logging
import tempfile
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import models
from src.utils.arg_parser import get_input_args
from src.utils.image_normalization import process_image, imshow
from flask import Flask, jsonify, request

# Check if Google Cloud Storage utilities are available
try:
    from src.utils.gcs_utils import is_gcs_path, download_from_gcs, parse_gcs_path
    HAS_GCS_SUPPORT = True
except ImportError:
    logging.warning("Google Cloud Storage utilities not available. GCS features will be disabled.")
    HAS_GCS_SUPPORT = False

# Check if MLflow is available for model loading
try:
    import mlflow.pyfunc
    HAS_MLFLOW_SUPPORT = True
except ImportError:
    logging.warning("MLflow not available. MLflow features will be disabled.")
    HAS_MLFLOW_SUPPORT = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Use environment variables with defaults for configuration
CAT_NAMES_PATH = os.environ.get('CAT_NAMES_PATH', os.path.join(PROJECT_ROOT, "configs/cat_to_name.json"))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(PROJECT_ROOT, "models"))

# GCS configuration
GCS_BUCKET = os.environ.get('GCS_BUCKET', None)  # Set this in your environment or docker-compose
if GCS_BUCKET and HAS_GCS_SUPPORT:
    logger.info(f"Google Cloud Storage bucket configured: {GCS_BUCKET}")
else:
    logger.info("Using local storage only for model files")

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
USE_MLFLOW_MODELS = os.environ.get('USE_MLFLOW_MODELS', 'false').lower() in ('true', '1', 'yes')

if HAS_MLFLOW_SUPPORT and USE_MLFLOW_MODELS:
    logger.info(f"MLflow configured with tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow model loading enabled")
else:
    logger.info("Using direct model loading (not using MLflow)")

# Log the configuration
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Category names path: {CAT_NAMES_PATH}")
logger.info(f"Models directory: {MODELS_DIR}")

with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

def process_base64_image(base64_image):
    
    # Decode the base64 string
    img_data = base64.b64decode(base64_image)
    img = Image.open(BytesIO(img_data))
    
    # Save to a temporary file
    temp_path = os.path.join(PROJECT_ROOT, "temp_image.jpg")
    img.save(temp_path)
    
    # Process using the existing function
    processed_img = process_image(temp_path)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return processed_img


def load_checkpoint(checkpoint_path):
    try:
        # Check if path is a GCS path and download if needed
        local_checkpoint_path = checkpoint_path
        if HAS_GCS_SUPPORT and is_gcs_path(checkpoint_path):
            logger.info(f"Downloading checkpoint from GCS: {checkpoint_path}")
            # Create a temporary file to download the checkpoint
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
                local_checkpoint_path = temp_file.name
            
            # Download the checkpoint from GCS
            download_from_gcs(checkpoint_path, local_checkpoint_path)
            logger.info(f"Downloaded checkpoint to {local_checkpoint_path}")
        
        # Load the checkpoint with CPU mapping for CUDA tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading checkpoint using device: {device}")
        
        # Use map_location to handle models saved on CUDA devices
        # Set weights_only=False to handle PyTorch 2.6+ security changes
        checkpoint = torch.load(local_checkpoint_path, map_location=device, weights_only=False)
        
        # Clean up temporary file if we downloaded from GCS
        if local_checkpoint_path != checkpoint_path and HAS_GCS_SUPPORT and is_gcs_path(checkpoint_path):
            os.remove(local_checkpoint_path)
            logger.info(f"Removed temporary checkpoint file: {local_checkpoint_path}")
        
        architecture = checkpoint.get("architecture", "vgg13")
        logger.info(f"Model architecture: {architecture}")

        if architecture == "vgg11":
            model = models.vgg11(weights=None)  # Weights=None, since we’ll load ours
        elif architecture == "vgg13":
            model = models.vgg13(weights=None)
        elif architecture == "vgg16":
            model = models.vgg16(weights=None)
        elif architecture == "vgg19":
            model = models.vgg19(weights=None)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        if 'classifier' in checkpoint:
            model.classifier = checkpoint['classifier']
        
        # Load the state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint.get('class_to_idx', {})
        
        # Optional: Print details
        epoch = checkpoint.get('epoch', 'Unknown')
        loss = checkpoint.get('loss', 'Unknown')
        logger.info(f"Checkpoint loaded from {checkpoint_path}: epoch {epoch}, loss {loss}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

# Function to load model from MLflow if enabled
def load_model_from_mlflow(run_id, model_name="PyTorchModel"):
    if not HAS_MLFLOW_SUPPORT:
        logger.warning("MLflow support not available. Cannot load model from MLflow.")
        return None
    
    try:
        logger.info(f"Loading model from MLflow run: {run_id}, model name: {model_name}")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        return None

# Function to get model path (local or GCS)
def get_model_path(model_name):
    # First check local path
    local_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(local_path):
        return local_path
    
    # If not found locally and GCS is configured, check GCS
    if HAS_GCS_SUPPORT and GCS_BUCKET:
        gcs_path = f"gs://{GCS_BUCKET}/models/{model_name}"
        return gcs_path
    
    # Default to local path if GCS not available
    return local_path

# Load models with error handling for containerized environment
try:
    # Check if we should use MLflow for model loading
    if HAS_MLFLOW_SUPPORT and USE_MLFLOW_MODELS:
        # MLflow run IDs for model versions (would be set via environment variables in production)
        MLFLOW_RUN_ID_V1 = os.environ.get('MLFLOW_RUN_ID_V1', None)
        MLFLOW_RUN_ID_V2 = os.environ.get('MLFLOW_RUN_ID_V2', None)
        
        if MLFLOW_RUN_ID_V1:
            logger.info(f"Loading model v1 from MLflow run: {MLFLOW_RUN_ID_V1}")
            model_v1 = load_model_from_mlflow(MLFLOW_RUN_ID_V1)
        else:
            logger.warning("No MLflow run ID for model v1. Falling back to direct checkpoint loading.")
            model_v1_path = get_model_path("model_checkpoint_v1.pth")
    # Only try to load model_v1 if we haven't already loaded it from MLflow
    if not (HAS_MLFLOW_SUPPORT and USE_MLFLOW_MODELS and MLFLOW_RUN_ID_V1 and 'model_v1' in locals()):
        logger.info(f"Loading model v1 from {model_v1_path}")
        model_v1 = load_checkpoint(model_v1_path)
        if model_v1 is None:
            logger.error(f"Failed to load model v1 from {model_v1_path}")
        else:
            logger.info(f"Model v1 loaded successfully")
    
    # Load model_v2 from MLflow if possible, otherwise from checkpoint
    if HAS_MLFLOW_SUPPORT and USE_MLFLOW_MODELS and MLFLOW_RUN_ID_V2:
        logger.info(f"Loading model v2 from MLflow run: {MLFLOW_RUN_ID_V2}")
        model_v2 = load_model_from_mlflow(MLFLOW_RUN_ID_V2)
    else:
        model_v2_path = get_model_path("model_checkpoint_v2.pth")
        logger.info(f"Loading model v2 from {model_v2_path}")
        model_v2 = load_checkpoint(model_v2_path)
    
    if model_v2 is None:
        logger.error(f"Failed to load model v2")
    else:
        logger.info(f"Model v2 loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")


def predict(image_data, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    
    # Convert to PyTorch tensor
    tensor_image = torch.from_numpy(image_data).type(torch.FloatTensor)
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    
    # Move model and tensor to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor_image = tensor_image.to(device)
    
    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        output = model(tensor_image)
    
    # Convert output to probabilities
    ps = torch.exp(output)
    
    # Get the top k probabilities and classes
    top_ps, top_indices = ps.topk(top_k, dim=1)
    
    # Move to CPU and convert to lists
    probs = top_ps.cpu().numpy().flatten().tolist()
    indices = top_indices.cpu().numpy().flatten().tolist()
    
    # Ensure class_to_idx exists
    if not hasattr(model, "class_to_idx"):
        raise AttributeError("Model does not have class_to_idx attribute.")
    
    # Invert class_to_idx mapping
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Map indices to actual class labels
    classes = [idx_to_class[idx] for idx in indices]
    
    # Get human-readable class names
    class_names = [cat_to_name.get(cls, cls) for cls in classes]
    
    # Format the results
    results = {
        "probabilities": probs,
        "classes": classes,
        "class_names": class_names
    }
    
    return results


@app.route('/flowers_classification_home', methods=['GET'])
def home():
    app_info = {
        "name": "Flowers Classification API",
        "description": "This API takes an image and returns the top K categories based on 102 Flowers Categories",
        "version": "v1.0",
        "endpoints": {
            "/flowers_classification_home": "Home Page",
            "/helth_status" : "Check APIs Health",
            "/v1/predict": "This version of the API is based on VGG19",
            "/v2/predict": "This version of the API is based on VGG13"
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
        }
    }
    
    return jsonify(app_info)

@app.route('/helth_status', methods=['GET'])
def helth_check():
    
    health = {
        "status": "UP",
        "message": "Flowers Classification is up and ready to recieve request"
    }
    return jsonify(health)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    logger.info("Received prediction request for model v1")
    
    if not request.is_json:
        logger.warning("Request does not contain JSON data")
        return jsonify({"error": "Request must contain JSON data"}), 400
    
    data = request.json
    
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
        logger.info(f"Processing image for prediction with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        results = predict(np_image, model_v1, top_k)
        
        logger.info("Prediction successful")
        return jsonify({
            "success": True,
            "prediction": results
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    if not request.is_json:
        return jsonify({"error": "Request must contain JSON data"}), 400
    
    data = request.json

    if 'image_data' not in data:
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    try:
        np_image = process_base64_image(base64_image)
        results = predict(np_image, model_v2, top_k)
        
        return jsonify({
            "success": True,
            "prediction": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')  # Use 0.0.0.0 to listen on all interfaces in container
    port = int(os.environ.get('FLASK_PORT', 9000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flowers Classification API on {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)