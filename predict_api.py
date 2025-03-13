import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import models
from src.utils.arg_parser import get_input_args
from src.utils.image_normalization import process_image, imshow
from flask import Flask, jsonify, request

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

CAT_NAMES_PATH = os.path.join(PROJECT_ROOT, "configs/cat_to_name.json")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

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
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path)
        architecture = checkpoint.get("architecture", "vgg13")
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
        print(f"Checkpoint loaded from {checkpoint_path}: epoch {epoch}, loss {loss}")
        
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

model_v1 = load_checkpoint(os.path.join(MODELS_DIR, "model_checkpoint_v1.pth"))
model_v2 = load_checkpoint(os.path.join(MODELS_DIR, "model_checkpoint_v2.pth"))


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
    
    if not request.is_json:
        return jsonify({"error": "Request must contain JSON data"}), 400
    
    data = request.json
    
    if 'image_data' not in data:
        return jsonify({"error": "Missing required field: image_data"}), 400
    
    base64_image = data['image_data']
    top_k = data.get('top_k', 5)
    
    try:
        np_image = process_base64_image(base64_image)
        results = predict(np_image, model_v1, top_k)
        
        return jsonify({
            "success": True,
            "prediction": results
        })
        
    except Exception as e:
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
    app.run(host='127.0.0.1', port=9000, debug=True)