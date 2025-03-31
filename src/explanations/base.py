"""
Base explanation functionality for flower classification model
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.segmentation import mark_boundaries

# Configure logging
logger = logging.getLogger(__name__)

# Get project root path - adjust for new location in src directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
