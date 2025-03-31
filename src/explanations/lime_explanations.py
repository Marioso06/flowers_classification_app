"""
LIME explanation generation for flower classification model
"""
import logging
import traceback
import numpy as np
import torch
from lime import lime_image

from src.utils.model_utils import predict
from src.explanations.base import create_explanation_visualization
from src.explanations.fallback import generate_fallback_explanation

# Configure logging
logger = logging.getLogger(__name__)

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
