"""
Image processing utilities for flower classification app
"""
import os
import base64
import logging
import numpy as np
from io import BytesIO
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Get project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def process_base64_image(base64_image):
    """Process a base64 encoded image into a format suitable for model input
    
    Args:
        base64_image: Base64 encoded string of image data
        
    Returns:
        np.array: Processed image in numpy array format
    """
    # Decode the base64 string
    img_data = base64.b64decode(base64_image)
    img = Image.open(BytesIO(img_data))
    
    # Save to a temporary file
    temp_path = os.path.join(PROJECT_ROOT, "temp_image.jpg")
    img.save(temp_path)
    
    # Use our standard image processing function
    from src.utils.image_normalization import process_image
    return process_image(temp_path)


def normalize_image_for_explanation(image_data):
    """Normalize image data to standard format for explanations
    
    Ensures image is in (H, W, 3) format with uint8 values 0-255
    
    Args:
        image_data: Input image data, usually from model preprocessing
        
    Returns:
        np.array: Normalized image in HWC format with uint8 values
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
