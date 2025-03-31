"""
SHAP explanation generation for flower classification model
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import shap
from skimage.transform import resize

from src.utils.model_utils import predict, get_model_wrapper
from src.explanations.base import create_explanation_visualization

# Configure logging
logger = logging.getLogger(__name__)

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
    class_shap_values = shap_values[class_idx_pos][0]  # [class_idx_pos] selects the class, [0] the first sample
    
    # Convert from CHW to HWC for visualization
    class_shap_values = np.transpose(class_shap_values, (1, 2, 0))
    
    # Aggregate SHAP values across color channels
    # This gives us a 2D heatmap indicating the importance of each pixel
    shap_heatmap = np.sum(np.abs(class_shap_values), axis=2)
    
    # Normalize for better visualization
    if shap_heatmap.max() > 0:
        shap_heatmap = shap_heatmap / shap_heatmap.max()
    
    # Return the visualization
    return create_explanation_visualization(
        original_image=original_image,
        heatmap=shap_heatmap,
        prediction_results=prediction_results,
        explanation_type="DeepSHAP",
        title2="SHAP Values",
        title3="SHAP Overlay",
        cmap="RdBu_r",
        alpha=0.7,
        third_panel_type="overlay"
    )
