"""
Fallback explanation methods for when primary explanation methods fail
"""
import os
import logging
import numpy as np
import torch
import traceback
from PIL import Image
from datetime import datetime
from torchvision import transforms

from src.utils.model_utils import predict
from src.utils.image_processing import normalize_image_for_explanation
from src.explanations.base import create_explanation_visualization

# Configure logging
logger = logging.getLogger(__name__)

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
        
        # Use our shared visualization function for consistent output
        return create_explanation_visualization(
            original_image=original_image,
            heatmap=saliency_map,
            prediction_results=prediction_results,
            explanation_type="SHAP",
            title2="Feature Importance",
            title3="Importance Overlay",
            cmap="RdBu_r",
            alpha=0.7,
            third_panel_type="overlay"
        )
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
