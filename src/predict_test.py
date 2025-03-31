import torch
import os
import numpy as np
import json
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
import mlflow.pytorch
from utils.image_normalization import process_image
from utils.arg_parser import get_input_args, load_config

# Define the root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Create Flask app
app = Flask(__name__)

class FlowerPredictor:
    def __init__(self, checkpoint_path=None, category_names=None, top_k=5):
        """
        Initialize the predictor with a model checkpoint
        
        Args:
            checkpoint_path: Path to the model checkpoint
            category_names: Path to the category names mapping file
            top_k: Number of top predictions to return
        """
        self.top_k = top_k
        self.model = None
        self.cat_to_name = {}
        
        # Load category mapping if provided
        if category_names and os.path.exists(category_names):
            with open(category_names, 'r') as f:
                self.cat_to_name = json.load(f)
        
        # Load the model if checkpoint path is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model from a checkpoint file
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Load the model architecture
            from torchvision import models
            architecture = checkpoint.get('architecture', 'vgg19')
            model = getattr(models, architecture)(pretrained=True)
            
            # Rebuild the classifier if it exists in the checkpoint
            if 'classifier' in checkpoint:
                model.classifier = checkpoint['classifier']
            
            # Load the state dictionaries
            model.load_state_dict(checkpoint['model_state_dict'])
            model.class_to_idx = checkpoint.get('class_to_idx', {})
            
            # Set model to evaluation mode
            model.eval()
            
            self.model = model
            print(f"Model loaded successfully from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def load_from_mlflow(self, run_id=None, model_uri=None):
        """
        Load a model from MLflow
        
        Args:
            run_id: MLflow run ID
            model_uri: MLflow model URI
        """
        try:
            if model_uri:
                self.model = mlflow.pytorch.load_model(model_uri)
            elif run_id:
                self.model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            else:
                # Load the latest model from the experiment
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name("Flowers Classification App")
                if experiment:
                    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
                    if runs:
                        latest_run = sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]
                        self.model = mlflow.pytorch.load_model(f"runs:/{latest_run.info.run_id}/model")
            
            if self.model:
                self.model.eval()
                print("Model loaded successfully from MLflow")
                return True
            return False
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return False
    
    def predict_image_path(self, image_path):
        """
        Predict the class of an image using a trained deep learning model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with probabilities and class names
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        # Process the image
        try:
            np_image = process_image(image_path)
            return self._predict(np_image)
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def predict_image_data(self, image_data):
        """
        Predict the class of an image from binary data
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary with probabilities and class names
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        try:
            # Convert binary data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Save to a temporary file
            temp_path = os.path.join(PROJECT_ROOT, "temp_image.jpg")
            image.save(temp_path)
            
            # Process the image
            np_image = process_image(temp_path)
            
            # Remove temporary file
            os.remove(temp_path)
            
            return self._predict(np_image)
        except Exception as e:
            return {"error": f"Error processing image data: {str(e)}"}
    
    def _predict(self, np_image):
        """
        Internal method to make predictions on a processed image
        
        Args:
            np_image: Processed numpy image array
            
        Returns:
            Dictionary with probabilities and class names
        """
        # Convert to PyTorch tensor
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        # Move model and tensor to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        tensor_image = tensor_image.to(device)
        
        # Make predictions
        with torch.no_grad():
            output = self.model(tensor_image)
        
        # Convert output to probabilities
        ps = torch.exp(output)
        
        # Get the top k probabilities and classes
        top_ps, top_indices = ps.topk(self.top_k, dim=1)
        
        # Move to CPU and convert to lists
        top_ps = top_ps.cpu().numpy().flatten().tolist()
        top_indices = top_indices.cpu().numpy().flatten().tolist()
        
        # Ensure class_to_idx exists
        if not hasattr(self.model, "class_to_idx"):
            return {"error": "Model does not have class_to_idx attribute"}
        
        # Invert class_to_idx mapping
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        
        # Map indices to actual class labels
        top_classes = [idx_to_class[idx] for idx in top_indices]
        
        # Map class labels to category names if available
        class_names = [self.cat_to_name.get(cls, cls) for cls in top_classes]
        
        # Create result dictionary
        result = {
            "probabilities": top_ps,
            "classes": top_classes,
            "class_names": class_names
        }
        
        return result


# Initialize predictor with default values
predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/load_model', methods=['POST'])
def load_model():
    """
    Load a model from a checkpoint file or MLflow
    
    Request body:
    {
        "checkpoint_path": "/path/to/checkpoint.pth",  # Optional
        "category_names": "/path/to/cat_to_name.json", # Optional
        "top_k": 5,                                    # Optional
        "mlflow_run_id": "run_id",                     # Optional
        "mlflow_model_uri": "model_uri"                # Optional
    }
    """
    global predictor
    
    data = request.json
    checkpoint_path = data.get('checkpoint_path')
    category_names = data.get('category_names')
    top_k = data.get('top_k', 5)
    mlflow_run_id = data.get('mlflow_run_id')
    mlflow_model_uri = data.get('mlflow_model_uri')
    
    # Create a new predictor
    predictor = FlowerPredictor(
        checkpoint_path=checkpoint_path,
        category_names=category_names,
        top_k=top_k
    )
    
    # Load from MLflow if specified
    if mlflow_run_id or mlflow_model_uri:
        success = predictor.load_from_mlflow(
            run_id=mlflow_run_id,
            model_uri=mlflow_model_uri
        )
        if success:
            return jsonify({"status": "Model loaded from MLflow successfully"})
        else:
            return jsonify({"status": "Failed to load model from MLflow"}), 500
    
    # Check if model was loaded from checkpoint
    if predictor.model:
        return jsonify({"status": "Model loaded successfully"})
    else:
        return jsonify({"status": "Model not loaded, check parameters"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of an image
    
    Request body:
    {
        "image_path": "/path/to/image.jpg"  # Optional, either this or image_data
        "image_data": "base64_encoded_image" # Optional, either this or image_path
    }
    """
    global predictor
    
    # Check if model is loaded
    if not predictor or not predictor.model:
        # Try to load default model from config
        config = load_config()
        predictor = FlowerPredictor(
            checkpoint_path=config.get("checkpoint"),
            category_names=config.get("category_names"),
            top_k=config.get("top_k", 5)
        )
        
        if not predictor.model:
            return jsonify({"error": "No model loaded. Use /load_model endpoint first."}), 400
    
    data = request.json
    
    # Check if image path is provided
    if 'image_path' in data:
        image_path = data['image_path']
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image file not found: {image_path}"}), 404
        
        result = predictor.predict_image_path(image_path)
        return jsonify(result)
    
    # Check if image data is provided
    elif 'image_data' in data:
        try:
            image_data = base64.b64decode(data['image_data'])
            result = predictor.predict_image_data(image_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Error processing image data: {str(e)}"}), 400
    
    else:
        return jsonify({"error": "No image provided. Include 'image_path' or 'image_data' in request."}), 400

@app.route('/predict/file', methods=['POST'])
def predict_file():
    """
    Predict the class of an image uploaded as a file
    """
    global predictor
    
    # Check if model is loaded
    if not predictor or not predictor.model:
        # Try to load default model from config
        config = load_config()
        predictor = FlowerPredictor(
            checkpoint_path=config.get("checkpoint"),
            category_names=config.get("category_names"),
            top_k=config.get("top_k", 5)
        )
        
        if not predictor.model:
            return jsonify({"error": "No model loaded. Use /load_model endpoint first."}), 400
    
    # Check if file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read file data
        image_data = file.read()
        result = predictor.predict_image_data(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 400

if __name__ == "__main__":
    # Parse arguments
    in_arg = get_input_args()
    
    # Initialize predictor with default values from config
    predictor = FlowerPredictor(
        checkpoint_path=in_arg.checkpoint,
        category_names=in_arg.category_names,
        top_k=in_arg.top_k
    )
    
    # Start Flask app
    app.run(host='127.0.0.1', port=9050, debug=True)
