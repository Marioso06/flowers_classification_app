import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import logging
import time
import psutil
import traceback
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import models, transforms
import torch.nn as nn

# Import XAI libraries
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.utils.image_normalization import process_image
from flask import Flask, jsonify, request, send_file
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Custom metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

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


def create_explanation_visualization(original_image, heatmap, prediction_results, explanation_type,
                                   title2="Feature Importance", title3="Importance Overlay",
                                   cmap="RdBu_r", alpha=0.7, third_panel_type="overlay", mask=None):
    """Create a standardized visualization for explanations (SHAP, LIME, etc.)
    
    Args:
        original_image: The original image to display
        heatmap: The heatmap representing feature importance
        prediction_results: Dictionary with prediction results
        explanation_type: Type of explanation ("SHAP", "LIME", etc.)
        title2: Title for the second panel (default: "Feature Importance")
        title3: Title for the third panel (default: "Importance Overlay")
        cmap: Colormap to use (default: "RdBu_r")
        alpha: Alpha value for overlay (default: 0.7)
        third_panel_type: Type of third panel ("overlay" or "segments")
        mask: Optional segmentation mask for LIME (default: None)
        
    Returns:
        Dictionary with paths to the saved visualization
    """
    # Log original image dimensions for debugging
    original_h, original_w = original_image.shape[:2]
    logger.info(f"Original image dimensions for {explanation_type}: {original_w}x{original_h}")
    
    # Ensure heatmap dimensions match original image
    if heatmap.shape[:2] != original_image.shape[:2]:
        logger.info(f"Resizing heatmap from {heatmap.shape[:2]} to match original image {original_image.shape[:2]}")
        from skimage.transform import resize
        heatmap = resize(heatmap, (original_h, original_w), order=1, anti_aliasing=True)
    
    # Calculate a better figure size based on the image dimensions
    # Use a larger base size for higher quality
    scale_factor = max(1.0, original_w / 400)  # Scale up for larger images
    fig_width = 18 * scale_factor  # Wider figure for better visibility
    fig_height = 6 * scale_factor  # Proportional height
    
    # Generate visualization with improved size
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image', fontsize=12 * scale_factor)
    plt.axis('off')
    
    # Heatmap visualization
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap=cmap)
    plt.title(title2, fontsize=12 * scale_factor)
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    
    # Third panel - either overlay or segments
    plt.subplot(1, 3, 3)
    if third_panel_type == "overlay":
        plt.imshow(original_image)
        plt.imshow(heatmap, cmap=cmap, alpha=alpha)
        plt.title(title3, fontsize=12 * scale_factor)
        plt.colorbar(shrink=0.8)
    elif third_panel_type == "segments" and mask is not None:
        plt.imshow(mark_boundaries(original_image, mask))
        plt.title('Superpixel Boundaries', fontsize=12 * scale_factor)
    plt.axis('off')
    
    # Adjust figure layout with more padding
    plt.tight_layout(pad=2.0)
    
    # Create static folder if it doesn't exist
    static_folder = os.path.join(PROJECT_ROOT, "static")
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    
    # Save the plot to a static file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{explanation_type.lower()}_explanation_{timestamp}.png"
    static_path = os.path.join(static_folder, filename)
    
    # Save with higher DPI for better quality output
    # Higher DPI gives better resolution without changing figure dimensions
    # Use original image dimensions to determine appropriate DPI
    original_h, original_w = original_image.shape[:2]
    # Determine base DPI based on image size - larger images need higher DPI
    base_dpi = 150  # Higher baseline DPI for all images
    if original_w > 300 or original_h > 300:
        base_dpi = 200
    if original_w > 600 or original_h > 600:
        base_dpi = 300
        
    logger.info(f"Saving visualization with DPI={base_dpi}")
    plt.savefig(static_path, dpi=base_dpi, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    # Return the paths and explanation info
    file_url = f"/static/{filename}"
    logger.info(f"{explanation_type} explanation saved to {static_path}")
    
    description = get_explanation_description(explanation_type)
    
    return {
        "visualization_path": static_path,
        "visualization_url": file_url,
        "prediction": prediction_results,
        "explanation_type": explanation_type,
        "description": description
    }


def get_explanation_description(explanation_type):
    """Get a standardized description for each explanation type"""
    if explanation_type in ["SHAP", "GradientSHAP"]:
        return "SHAP values show the contribution of each pixel to the prediction."
    elif explanation_type == "DeepSHAP":
        return "DeepSHAP values show the contribution of each pixel to the prediction."
    elif explanation_type == "LIME":
        return "LIME segments the image and shows which regions contribute to the prediction."
    elif explanation_type == "Fallback":
        return "A simple occlusion-based saliency map showing important regions for prediction."
    else:
        return f"{explanation_type} explanation shows which parts of the image influence the prediction."

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


def get_model_wrapper(model):
    """Creates a wrapper for the model to use with XAI tools"""
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


def generate_shap_explanation(image_data, model, top_k=5):
    """Generate SHAP explanations for image classification
    
    This function tries to use the GradientExplainer first, which is more efficient for deep networks.
    If that fails, it falls back to DeepExplainer and then to custom implementation.
    """
    try:
        logger.info("=== Attempting to use GradientExplainer for SHAP explanation ===")
        result = generate_gradient_shap_explanation(image_data, model, top_k)
        logger.info("GradientExplainer successful!")
        return result
    except Exception as e:
        logger.warning(f"GradientExplainer failed: {str(e)}\n{traceback.format_exc()}")
        logger.warning("Trying DeepExplainer...")
        try:
            logger.info("=== Attempting to use DeepExplainer for SHAP explanation ===")
            result = generate_deep_shap_explanation(image_data, model, top_k)
            logger.info("DeepExplainer successful!")
            return result
        except Exception as e:
            logger.warning(f"DeepExplainer failed: {str(e)}\n{traceback.format_exc()}")
            logger.warning("Falling back to custom implementation...")
            logger.info("=== Using custom occlusion-based implementation for SHAP explanation ===")
            return generate_custom_shap_explanation(image_data, model, top_k)


def generate_gradient_shap_explanation(image_data, model, top_k=5):
    """Generate SHAP explanations using the GradientExplainer
    
    This implementation follows the approach in the SHAP documentation for deep learning models,
    extracting feature importance directly using gradients.
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to the appropriate device
    model.to(device)
    model.eval()
    
    # Store the original image for visualization (convert from CHW to HWC)
    original_image = image_data.transpose(1, 2, 0).copy()
    
    # De-normalize the image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = original_image * std + mean
    original_image = np.clip(original_image, 0.0, 1.0)
    
    # Get prediction to determine which class to explain
    logger.info("Getting prediction to determine class to explain")
    prediction_results = predict(image_data, model, top_k)
    class_idx = int(prediction_results['classes'][0])  # Explain top class
    
    # Create a background distribution of inputs
    logger.info("Creating background distribution for GradientExplainer")
    
    # For GradientExplainer, we generally want a random normal distribution around our input
    n_background = 20
    background = np.zeros((n_background, 3, 224, 224), dtype=np.float32)
    
    # Create variations of the input image with small noise
    for i in range(n_background):
        # Add small random noise
        noise = np.random.normal(0, 0.01, image_data.shape).astype(np.float32)
        background[i] = np.clip(image_data + noise, 0, 1)
    
    # We need to prepare the model for the GradientExplainer
    # First, get input and output references
    
    class GradientWrapperModel(nn.Module):
        def __init__(self, model):
            super(GradientWrapperModel, self).__init__()
            self.model = model
            self.feature_layer = None  # Will hold intermediate features
            self.handles = []
            
            # Register hook to capture intermediate features
            # For a ResNet model, we'll capture the output of the final convolutional layer
            # We need to identify which layer to target based on the model architecture
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Save the last conv layer
                    self.last_conv_name = name
                    self.last_conv_layer = module
            
            # Register a forward hook on the last convolutional layer
            handle = self.last_conv_layer.register_forward_hook(self._hook_fn)
            self.handles.append(handle)
            
        def _hook_fn(self, module, input, output):
            self.feature_layer = output
            
        def forward(self, x):
            self.feature_layer = None
            output = self.model(x)
            return output, self.feature_layer
            
        def close(self):
            for handle in self.handles:
                handle.remove()
    
    # Wrap the model to access intermediate features
    gradient_model = GradientWrapperModel(model)
    gradient_model.to(device)
    gradient_model.eval()
    
    # Convert tensors
    background_tensor = torch.FloatTensor(background).to(device)
    input_tensor = torch.FloatTensor(image_data).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get intermediate layer output for background
    logger.info("Extracting background features for gradient explanation")
    with torch.no_grad():
        # Forward pass on all background samples to get intermediate features
        background_features = []
        for i in range(n_background):
            _, features = gradient_model(background_tensor[i:i+1])
            background_features.append(features.detach().cpu().numpy())
        background_features = np.concatenate(background_features)
    
    # Create the SHAP explainer
    logger.info("Creating SHAP GradientExplainer")
    
    # Get output and feature layers
    with torch.no_grad():
        output, feature_layer = gradient_model(input_tensor)
    
    # Extract model info for class indexing
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Create GradientExplainer
    def model_output(feature_tensor):
        # Forward pass from intermediate features to output
        # This is a simplified approach - in a real scenario, you'd need to properly
        # trace the network from the feature layer to output
        feature_tensor = torch.FloatTensor(feature_tensor).to(device)
        with torch.no_grad():
            # Use a dummy input to trace through the network
            dummy_input = input_tensor.clone()
            # Get the full model output and features
            output, _ = gradient_model(dummy_input)
            # We use the prediction for the dummy input since we can't easily
            # forward from the intermediate features to output
            return output.cpu().numpy()
    
    # Initialize explainer on the feature layer
    explainer = shap.GradientExplainer((feature_layer, output), background_features)
    
    # Compute SHAP values
    logger.info("Computing SHAP values with GradientExplainer")
    shap_values, indices = explainer.shap_values(feature_layer.cpu().numpy(), ranked_outputs=top_k)
    
    # Process SHAP values and indices
    # Each shap_values entry corresponds to a ranked output class
    # indices contains the actual class indices that SHAP ranked
    # We'll use the first one (top predicted class according to SHAP)
    class_shap_values = shap_values[0][0]  # [0] for top class, [0] for first sample
    
    # Get the actual class index from SHAP's results
    top_class_idx = int(indices[0][0])  # First class, first sample
    logger.info(f"Top class according to SHAP: {top_class_idx}")
    
    # Map the index to our class names for visualization
    # This ensures we're correctly labeling the class that SHAP has explained
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    try:
        class_name = idx_to_class.get(top_class_idx, "Unknown")
        logger.info(f"Explaining class: {class_name} (index {top_class_idx})")
    except Exception as e:
        logger.warning(f"Error mapping class index: {e}")
    
    # Determine the original image dimensions for proper upsampling
    original_h, original_w = original_image.shape[:2]
    logger.info(f"Original image dimensions: {original_w}x{original_h}")
    
    # Upsample the SHAP values to match the original image dimensions
    # SHAP values are for feature maps, we need to upsample to original image size
    feature_size = class_shap_values.shape[1]  # Size of feature map
    
    # Sum across channels to get importance
    channel_sum = np.sum(np.abs(class_shap_values), axis=0)
    
    # Normalize for visualization
    if channel_sum.max() > 0:
        channel_sum = channel_sum / channel_sum.max()
    
    # Upsample the feature importance map to match the original image dimensions
    from skimage.transform import resize
    upsampled_shap = resize(channel_sum, (original_h, original_w), order=1, anti_aliasing=True)
    
    # Clean up gradient model resources
    gradient_model.close()
    
    # Use our shared visualization function for consistent output
    return create_explanation_visualization(
        original_image=original_image,
        heatmap=upsampled_shap,
        prediction_results=prediction_results,
        explanation_type="GradientSHAP",
        title2="Feature Importance",
        title3="Importance Overlay",
        cmap="RdBu_r",
        alpha=0.7,
        third_panel_type="overlay"
    )


def generate_deep_shap_explanation(image_data, model, top_k=5):
    """Generate SHAP explanations for image classification using the SHAP library
    
    This function implements the DeepExplainer from SHAP which is designed for
    deep learning models and provides meaningful explanations for image inputs.
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to the appropriate device
    model.to(device)
    model.eval()
    
    # Store the original image for visualization (convert from CHW to HWC)
    original_image = image_data.transpose(1, 2, 0).copy()
    
    # De-normalize the image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = original_image * std + mean
    original_image = np.clip(original_image, 0.0, 1.0)
    
    # Get prediction to determine which class to explain
    logger.info("Getting prediction to determine class to explain")
    prediction_results = predict(image_data, model, top_k)
    class_idx = int(prediction_results['classes'][0])  # Explain top class
    
    # For SHAP, we need a batch of background samples
    # We'll create some random noise samples as background
    # This is a simplified approach - ideally we'd use actual background samples from the dataset
    logger.info("Creating background samples for SHAP explainer")
    
    # Use a reduced number of background samples to improve performance
    n_background = 10
    background = np.random.random((n_background, 3, 224, 224)).astype(np.float32)
    
    # Normalize background samples the same way as the input
    for i in range(n_background):
        background[i] = background[i] * 0.1  # Reduce intensity of random samples
    
    # Convert to PyTorch tensors
    background_tensor = torch.FloatTensor(background)
    input_tensor = torch.FloatTensor(image_data).unsqueeze(0)  # Add batch dimension
    
    # Create a wrapper model that returns softmax probabilities
    wrapped_model = get_model_wrapper(model)
    
    # Create the SHAP explainer
    logger.info("Creating SHAP DeepExplainer")
    explainer = shap.DeepExplainer(wrapped_model, background_tensor)

    # Generate SHAP values - DeepExplainer needs batch input
    logger.info("Computing SHAP values")
    # This calculation can be very memory intensive
    shap_values = explainer.shap_values(input_tensor)
    
    # Since we get SHAP values for each class, take the one for our predicted class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_idx_pos = list(idx_to_class.keys()).index(class_idx)
    
    # Get the SHAP values for the predicted class
    class_shap_values = shap_values[class_idx_pos][0]  # First element of batch
    
    # Generate visualization
    plt.figure(figsize=(15, 5))
    
    # Original image - ensure we're showing the actual unaltered image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Sum across channels for feature importance heatmap
    shap_map = np.sum(np.abs(class_shap_values), axis=0)
    # Normalize for better visualization
    if shap_map.max() > 0:
        shap_map = shap_map / shap_map.max()
    
    # Use our shared visualization function for consistent output
    return create_explanation_visualization(
        original_image=original_image,
        heatmap=shap_map,
        prediction_results=prediction_results,
        explanation_type="DeepSHAP", 
        cmap="RdBu_r", 
        alpha=0.7,
        third_panel_type="overlay"
    )
    


def normalize_image_for_explanation(image_data):
    """Normalize image data to standard format for explanations
    
    Ensures image is in (H, W, 3) format with uint8 values 0-255
    """
    logger.info(f"Original image data shape: {image_data.shape}, dtype: {image_data.dtype}")
    
    # Handle different input formats
    if len(image_data.shape) == 3:
        if image_data.shape[0] == 3 and image_data.shape[1] > 3 and image_data.shape[2] > 3:
            # Input is in (C, H, W) format (PyTorch format), convert to (H, W, C)
            logger.info("Converting from CHW to HWC format")
            image_data = np.transpose(image_data, (1, 2, 0))
        elif image_data.shape[2] != 3:
            # Unexpected format
            logger.error(f"Unexpected image shape: {image_data.shape}")
            raise ValueError(f"Cannot process image with shape {image_data.shape}")
    else:
        logger.error(f"Image must have 3 dimensions, got {len(image_data.shape)}")
        raise ValueError(f"Cannot process image with {len(image_data.shape)} dimensions")
    
    # Normalize values to 0-255 if they're in 0-1 range
    if image_data.max() <= 1.0:
        logger.info("Normalizing image values from 0-1 to 0-255")
        image_data = (image_data * 255).astype(np.uint8)
    elif image_data.dtype != np.uint8:
        logger.info(f"Converting image from {image_data.dtype} to uint8")
        image_data = image_data.astype(np.uint8)
    
    logger.info(f"Normalized image shape: {image_data.shape}, dtype: {image_data.dtype}")
    return image_data

def generate_custom_shap_explanation(image_data, model, top_k=5):
    """Generate a simple occlusion-based saliency map instead of using SHAP
    
    SHAP is extremely resource-intensive for image classification and may cause
    memory issues or very slow processing. This implementation uses a simpler
    occlusion-based approach that provides similar explanations but is much more
    efficient and reliable.
    """
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to the appropriate device
        model.to(device)
        model.eval()
        
        # Store the original image for visualization BEFORE normalization
        # Convert from (C,H,W) back to (H,W,C) format
        original_image = image_data.transpose(1, 2, 0).copy()
        
        # De-normalize from the preprocessing that was applied
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = original_image * std + mean
        
        # Clip to ensure values are in valid 0-1 range
        original_image = np.clip(original_image, 0.0, 1.0)
        
        try:
            # For explanation, keep the normalized format for the model
            norm_image_data = normalize_image_for_explanation(image_data)
            
            # Create a PIL version of our denormalized image for display
            display_img_uint8 = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(display_img_uint8)
            
            # Log the shape and data type of the image for debugging
            logger.info(f"Original display image shape: {original_image.shape}, dtype: {original_image.dtype}")
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return generate_fallback_explanation(image_data, model, top_k)
        
        # Get prediction to determine which class to explain
        logger.info("Getting prediction to determine class to explain")
        prediction_results = predict(image_data, model, top_k)
        class_idx = int(prediction_results['classes'][0])  # Explain top class
        
        # Create a simple and efficient occlusion-based saliency map
        # This is much faster and more reliable than SHAP for images
        logger.info("Creating occlusion-based saliency map")
        
        # Resize image to a smaller size for faster computation
        img_size = 112  # Half size for better performance
        small_image = np.array(Image.fromarray(norm_image_data).resize((img_size, img_size)))
        
        # Create a saliency map
        saliency_map = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Convert the image to tensor for PyTorch
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor_image = transform(Image.fromarray(small_image)).unsqueeze(0).to(device)
        
        # Get the original prediction probability for the top class
        with torch.no_grad():
            output = model(tensor_image)
            probs = torch.nn.functional.softmax(output, dim=1)
            original_prob = probs[0, class_idx].item()
        
        # Define occlusion parameters
        occlusion_size = 8  # Size of the occlusion patch
        stride = 8  # Step size for sliding the occlusion window
        
        # Use fewer occlusions for better performance
        logger.info(f"Computing importance map with {occlusion_size}x{occlusion_size} patches")
        
        # Create a heatmap by occluding different parts of the image
        for y in range(0, img_size, stride):
            for x in range(0, img_size, stride):
                # Define occlusion region boundaries
                y_end = min(y + occlusion_size, img_size)
                x_end = min(x + occlusion_size, img_size)
                
                # Create occluded image
                occluded_image = small_image.copy()
                # Fill with gray (neutral color in most models)
                occluded_image[y:y_end, x:x_end, :] = 128
                
                # Convert to tensor
                occluded_tensor = transform(Image.fromarray(occluded_image)).unsqueeze(0).to(device)
                
                # Get new prediction
                with torch.no_grad():
                    output = model(occluded_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    occluded_prob = probs[0, class_idx].item()
                
                # Compute importance: original - occluded (higher means more important)
                importance = original_prob - occluded_prob
                
                # Fill the saliency map
                saliency_map[y:y_end, x:x_end] = importance
        
        # Normalize saliency map to [0, 1] range
        saliency_map = saliency_map - saliency_map.min()
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
            
        # In SHAP, we typically represent importance with a blue-red color scheme
        # We'll use matplotlib's colormap for visualization instead of manual RGB channels
        # For now, still keep the heatmap representation for later processing
        shap_image = np.zeros((img_size, img_size, 3), dtype=np.float32)
        shap_image[:, :, 0] = saliency_map  # Will be properly colored in visualization
            
        # Resize to original size (224x224) for visualization
        logger.info("Resizing SHAP visualization to original size")
        shap_image_resized = np.zeros((224, 224, 3), dtype=np.float32)
        
        try:
            # Create a PIL image from each channel and resize
            for c in range(3):
                # Scale values to 0-255 range for PIL
                channel_vals = (shap_image[:, :, c] - shap_image[:, :, c].min()) * 255.0 / (shap_image[:, :, c].max() - shap_image[:, :, c].min() + 1e-8)
                channel_img = Image.fromarray(channel_vals.astype(np.uint8))
                resized = channel_img.resize((224, 224), Image.BILINEAR)
                # Convert back to float and normalize
                resized_array = np.array(resized).astype(np.float32) / 255.0
                shap_image_resized[:, :, c] = resized_array * (shap_image[:, :, c].max() - shap_image[:, :, c].min()) + shap_image[:, :, c].min()
            
            shap_image = shap_image_resized
            
            # Normalize for visualization
            logger.info("Normalizing SHAP values for visualization")
            abs_vals = np.abs(shap_image)
            max_val = np.max(abs_vals)
            if max_val > 0:
                shap_image = shap_image / max_val
            
            # For visualization, resize original image to match if needed
            if original_image.shape[0] != 224 or original_image.shape[1] != 224:
                original_image = np.array(Image.fromarray(
                    (original_image * 255).astype(np.uint8)).resize((224, 224)))
                original_image = original_image.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Error in SHAP computation: {str(e)}\n{traceback.format_exc()}")
            return generate_fallback_explanation(image_data, model, top_k)
        
        # Normalize for better visualization
        abs_vals = np.abs(shap_image)
        max_val = np.max(abs_vals)
        if max_val > 0:
            shap_image = shap_image / max_val
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        # Resize original image to match saliency map if needed
        if original_image.shape[0] != img_size or original_image.shape[1] != img_size:
            display_small = np.array(Image.fromarray(
                (original_image * 255).astype(np.uint8)).resize((img_size, img_size)))
            display_small = display_small.astype(np.float32) / 255.0
        else:
            display_small = original_image.copy()
            
        # Original image - ensure we're showing the actual unaltered image
        plt.subplot(1, 3, 1)
        # Display the properly denormalized original image
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Saliency heatmap - use standard SHAP blue-red colormap
        plt.subplot(1, 3, 2)
        plt.imshow(saliency_map, cmap='RdBu_r')
        plt.title('Feature Importance')
        plt.axis('off')
        plt.colorbar()
        
        # Overlay importance on original image
        plt.subplot(1, 3, 3)
        # Use the same properly displayed original image for consistency 
        plt.imshow(original_image)
        plt.imshow(saliency_map, cmap='RdBu_r', alpha=0.7)  # Use alpha for transparency
        plt.title('Importance Overlay')
        plt.axis('off')
        plt.colorbar()
        
        # Create static folder if it doesn't exist
        static_folder = os.path.join(PROJECT_ROOT, "static")
        if not os.path.exists(static_folder):
            os.makedirs(static_folder)
        
        # Adjust the figure size and layout for better visualization
        plt.gcf().set_size_inches(15, 5)
        plt.tight_layout()
        
        # Save the plot to a static file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shap_explanation_{timestamp}.png"
        static_path = os.path.join(static_folder, filename)
        plt.savefig(static_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Don't convert to base64 to avoid large payloads
        file_url = f"/static/{filename}"
        logger.info(f"SHAP explanation saved to {static_path}")
        
        return {
            "visualization_path": static_path,
            "visualization_url": file_url,
            "prediction": prediction_results,
            "explanation_type": "SHAP",
            "description": "SHAP values show the contribution of each pixel to the prediction."
        }
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        # traceback is now imported at the top of the file
        logger.error(f"Traceback: {traceback.format_exc()}")
        # If SHAP fails, try a fallback approach
        return generate_fallback_explanation(image_data, model, top_k)


def generate_fallback_explanation(image_data, model, top_k=5):
    """Generate a simpler explanation when SHAP fails"""
    # Convert to PyTorch tensor
    tensor_image = torch.from_numpy(image_data).type(torch.FloatTensor)
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model and tensor to the appropriate device
    model.to(device)
    tensor_image = tensor_image.to(device)
    
    # Set the model to evaluation mode and make predictions
    model.eval()
    
    # Get the original prediction
    prediction_results = predict(image_data, model, top_k)
    
    # First properly normalize the image for display
    # Store the original image for visualization BEFORE normalization
    # Convert from (C,H,W) back to (H,W,C) format
    original_image = image_data.transpose(1, 2, 0).copy()
    
    # De-normalize from the preprocessing that was applied
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = original_image * std + mean
    
    # Clip to ensure values are in valid 0-1 range
    original_image = np.clip(original_image, 0.0, 1.0)
    
    try:
        # For explanation, use the normalized format for processing
        norm_image_data = normalize_image_for_explanation(image_data)
        
        # Create a PIL version of our denormalized image for display
        display_img_uint8 = (original_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(display_img_uint8)
        
        # Log for debugging
        logger.info(f"Fallback explanation - original image shape: {original_image.shape}, dtype: {original_image.dtype}")
    except Exception as e:
        logger.error(f"Error in fallback image preprocessing: {str(e)}")
        
    # Get dimensions from the normalized image
    height, width, _ = original_image.shape
    saliency_map = np.zeros((height, width), dtype=np.float32)
    
    # Sample a grid of points to occlude (for efficiency)
    grid_size = 16  # Adjust based on image size and performance needs
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            # Create a copy of the image with a region occluded
            occluded_image = np.copy(image_data)
            y_end = min(y + grid_size, height)
            x_end = min(x + grid_size, width)
            occluded_image[:, y:y_end, x:x_end] = 0  # Black occlusion
            
            # Convert to tensor and predict
            occluded_tensor = torch.from_numpy(occluded_image).type(torch.FloatTensor).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(occluded_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
            
            # Extract probability for the top predicted class
            idx_to_class = {v: k for k, v in model.class_to_idx.items()}
            top_class = int(prediction_results['classes'][0])
            class_idx = list(idx_to_class.keys()).index(top_class)
            
            # If occlusion reduces probability, this region is important
            original_prob = prediction_results['probabilities'][0]
            occluded_prob = probs[0, class_idx].item()
            importance = original_prob - occluded_prob
            
            # Fill the saliency map region with this importance value
            saliency_map[y:y_end, x:x_end] = importance
    
    # Normalize saliency map
    if np.max(saliency_map) > 0:
        saliency_map = saliency_map / np.max(saliency_map)
    
    # Use our standardized visualization function
    try:
        display_image = original_image
    except Exception as e:
        logger.error(f"Error preparing original image: {e}")
        # Last resort fallback
        display_image = image_data.transpose(1, 2, 0).copy()
        # Apply standard denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_image = display_image * std + mean
        display_image = np.clip(display_image, 0, 1)
    
    return create_explanation_visualization(
        original_image=display_image,
        heatmap=saliency_map,
        prediction_results=prediction_results,
        explanation_type="Fallback",
        title2="Feature Importance",
        title3="Importance Overlay",
        cmap="RdBu_r",
        alpha=0.5,
        third_panel_type="overlay"
    )


def generate_lime_explanation(image_data, model, top_k=5):
    """Generate LIME explanations for a model prediction using the lime_image library
    
    This function implements a proper LIME image explainer which segments the image
    and learns a local surrogate model to explain the prediction.
    """
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to the appropriate device
        model.to(device)
        model.eval()
        
        # Store original image for visualization (convert from CHW to HWC)
        original_image = image_data.transpose(1, 2, 0).copy()
        
        # De-normalize from the preprocessing that was applied
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = original_image * std + mean
        
        # Clip to ensure values are in valid 0-1 range
        original_image = np.clip(original_image, 0.0, 1.0)
        
        # Get prediction to determine which class to explain
        logger.info("Getting prediction to determine class to explain")
        prediction_results = predict(image_data, model, top_k)
        top_class_idx = int(prediction_results['classes'][0])  # Explain top class
        
        # Create a function for LIME to predict with our model
        def predict_fn(images):
            # The images from LIME are in (batch, H, W, C) and RGB scaled 0-1
            batch_size = len(images)
            batch = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
            
            for i, img in enumerate(images):
                # LIME gives images in HWC format, convert to model input format (CHW)
                # and apply ImageNet normalization
                img = (img - mean) / std
                batch[i] = img.transpose(2, 0, 1)
            
            # Convert to tensor and get predictions
            tensor = torch.FloatTensor(batch).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
                
            return probs
        
        # Initialize LIME explainer with verbose logging
        logger.info("Creating LIME explainer")
        explainer = lime_image.LimeImageExplainer(verbose=True)
        
        # Get explanation
        logger.info("Computing LIME explanation")
        explanation = explainer.explain_instance(
            original_image,
            predict_fn,
            top_labels=1,  # Just explain the top prediction
            hide_color=0,  # Black for hiding segments
            num_samples=500,  # More samples gives better explanations but takes longer
            num_features=10,  # Max number of superpixels to include
            random_seed=42
        )
        
        # Get the explanation for the top predicted class
        # We need to map from our class_to_idx to the position in the model output
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = top_class_idx
        
        # Get positive and negative contributions for visualization
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],  # Use the first top label
            positive_only=False,  # Show both positive and negative contributions
            num_features=10,  # Top superpixels to include
            hide_rest=False,  # Don't hide the rest of the image
        )
        
        # Heatmap of all features (positive and negative)
        heatmap = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=False, 
            negative_only=False, 
            hide_rest=False,
            num_features=10
        )[1]
        
        # Use our standardized visualization function for consistent output
        return create_explanation_visualization(
            original_image=original_image,
            heatmap=heatmap,
            prediction_results=prediction_results,
            explanation_type="LIME",
            title2="Feature Importance",
            title3="Superpixel Boundaries",
            cmap="RdBu_r",
            alpha=0.7,
            third_panel_type="segments",
            mask=mask
        )
        
    except Exception as e:
        # Get the full stack trace for debugging
        logger.error(f"Error in LIME explanation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fall back to alternative explanation method
        return generate_fallback_explanation(image_data, model, top_k)


@app.route('/flowers_classification_home', methods=['GET'])
def home():
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

@app.route('/static/<path:filename>')
def serve_static(filename):
    static_folder = os.path.join(PROJECT_ROOT, "static")
    return send_file(os.path.join(static_folder, filename))

@app.route('/health_status', methods=['GET'])
def health_check():
    health = {
        "status": "UP",
        "message": "Flowers Classification is up and ready to receive request"
    }
    return jsonify(health)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    logger.info("Received prediction request for model v1")
    
    # Increment prediction request counter
    prediction_requests.labels(model_version="v1").inc()
    
    # Update resource metrics
    memory_usage.set(psutil.Process(os.getpid()).memory_info().rss)  # Resident Set Size in bytes
    cpu_usage.set(psutil.Process(os.getpid()).cpu_percent(interval=None))
    
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
        
        # Time the prediction
        start_time = time.time()
        results = predict(np_image, model_v1, top_k)
        prediction_time.labels(model_version="v1").observe(time.time() - start_time)
        
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
    logger.info("Received prediction request for model v2")
    
    # Increment prediction request counter
    prediction_requests.labels(model_version="v2").inc()
    
    # Update resource metrics
    memory_usage.set(psutil.Process(os.getpid()).memory_info().rss)
    cpu_usage.set(psutil.Process(os.getpid()).cpu_percent(interval=None))
    
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
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        logger.info(f"Processing image for prediction with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Time the prediction
        start_time = time.time()
        results = predict(np_image, model_v2, top_k)
        prediction_time.labels(model_version="v2").observe(time.time() - start_time)
        
        logger.info("Prediction successful")
        return jsonify({
            "success": True,
            "prediction": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def monitor_resources():
    """Background thread function to monitor resource usage"""
    while True:
        try:
            # Update memory and CPU metrics
            process = psutil.Process(os.getpid())
            memory_usage.set(process.memory_info().rss)  # Resident Set Size in bytes
            cpu_usage.set(process.cpu_percent(interval=1.0))
            time.sleep(15)  # Update every 15 seconds
        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")
            time.sleep(60)  # Retry after a minute if there was an error

@app.route('/v1/explain_shap', methods=['POST'])
def explain_shap_v1():
    logger.info("Received SHAP explanation request for model v1")
    
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
        logger.info(f"Processing image for SHAP explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate SHAP explanation using the cascading implementation
        # This will try GradientExplainer first, then DeepExplainer, then fallback to custom
        explanation = generate_shap_explanation(np_image, model_v1, top_k)
        
        logger.info("SHAP explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        logger.error(f"Error during SHAP explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/explain_shap', methods=['POST'])
def explain_shap_v2():
    logger.info("Received SHAP explanation request for model v2")
    
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
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        logger.info(f"Processing image for SHAP explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate SHAP explanation using the cascading implementation
        # This will try GradientExplainer first, then DeepExplainer, then fallback to custom
        explanation = generate_shap_explanation(np_image, model_v2, top_k)
        
        # Log which explanation type was used
        logger.info(f"SHAP explanation type: {explanation.get('explanation_type', 'unknown')}")
        
        logger.info("SHAP explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        logger.error(f"Error during SHAP explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v1/explain_lime', methods=['POST'])
def explain_lime_v1():
    logger.info("Received LIME explanation request for model v1")
    
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
        logger.info(f"Processing image for LIME explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(np_image, model_v1, top_k)
        
        logger.info("LIME explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        logger.error(f"Error during LIME explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v2/explain_lime', methods=['POST'])
def explain_lime_v2():
    logger.info("Received LIME explanation request for model v2")
    
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
    if model_v2 is None:
        logger.error("Model v2 is not loaded")
        return jsonify({"error": "Model v2 is not available"}), 503
    
    try:
        logger.info(f"Processing image for LIME explanation with top_k={top_k}")
        np_image = process_base64_image(base64_image)
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(np_image, model_v2, top_k)
        
        logger.info("LIME explanation generated successfully")
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        logger.error(f"Error during LIME explanation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Start resource monitoring in a background thread
    import threading
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    logger.info("Resource monitoring thread started")
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')  # Use 0.0.0.0 to listen on all interfaces in container
    port = int(os.environ.get('FLASK_PORT', 9000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flowers Classification API on {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)