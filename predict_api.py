import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import logging
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import models
from src.utils.arg_parser import get_input_args
from src.utils.image_normalization import process_image, imshow
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Use environment variables with defaults for configuration
CAT_NAMES_PATH = os.environ.get('CAT_NAMES_PATH', os.path.join(PROJECT_ROOT, "configs/cat_to_name.json"))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(PROJECT_ROOT, "models"))

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
        # Load the checkpoint with CPU mapping for CUDA tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading checkpoint using device: {device}")
        
        # Use map_location to handle models saved on CUDA devices
        # Set weights_only=False to handle PyTorch 2.6+ security changes
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
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

# Load models with error handling for containerized environment
try:
    model_v1_path = os.path.join(MODELS_DIR, "model_checkpoint_v1.pth")
    logger.info(f"Loading model v1 from {model_v1_path}")
    model_v1 = load_checkpoint(model_v1_path)
    if model_v1 is None:
        logger.error(f"Failed to load model v1 from {model_v1_path}")
    else:
        logger.info(f"Model v1 loaded successfully")
        
    model_v2_path = os.path.join(MODELS_DIR, "model_checkpoint_v2.pth")
    logger.info(f"Loading model v2 from {model_v2_path}")
    model_v2 = load_checkpoint(model_v2_path)
    if model_v2 is None:
        logger.error(f"Failed to load model v2 from {model_v2_path}")
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