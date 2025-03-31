"""
Model utility functions for flower classification
"""
import os
import json
import logging
import torch
import torch.nn as nn
from torchvision import models

# Configure logging
logger = logging.getLogger(__name__)

# Get project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(PROJECT_ROOT, "models"))
CAT_NAMES_PATH = os.environ.get('CAT_NAMES_PATH', os.path.join(PROJECT_ROOT, "configs/cat_to_name.json"))

# Load category names
with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from a file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Loaded PyTorch model
    """
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
            model = models.vgg11(weights=None)  # Weights=None, since we'll load ours
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


def predict(image_data, model, top_k=5):
    """
    Predict the class of an image using a trained deep learning model
    
    Args:
        image_data: Preprocessed image data as numpy array
        model: PyTorch model to use for prediction
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary containing probabilities, class indices, and class names
    """
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


def get_model_wrapper(model):
    """
    Creates a wrapper for the model to use with XAI tools
    
    Args:
        model: PyTorch model to wrap
        
    Returns:
        ModelWrapper: Wrapped model with additional methods
    """
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.device = next(model.parameters()).device
            self.class_to_idx = model.class_to_idx
            
        def forward(self, x):
            # Ensure input is on the correct device
            x = x.to(self.device)
            return self.model(x)
        
        def predict_proba(self, x):
            # Method for scikit-learn compatibility (used by SHAP)
            with torch.no_grad():
                output = self.forward(torch.FloatTensor(x))
                return torch.nn.functional.softmax(output, dim=1).cpu().numpy()
    
    return ModelWrapper(model)


# Function to load the models at module import time
def load_models():
    """
    Load both model versions from checkpoint files
    
    Returns:
        tuple: (model_v1, model_v2) - the loaded models
    """
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
            
        return model_v1, model_v2
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None
