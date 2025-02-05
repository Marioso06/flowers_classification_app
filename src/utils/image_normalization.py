from PIL import Image    
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    pil_image = Image.open(image_path)
    
    # Resize the image
    pil_image = pil_image.resize((256, 256))
    
    # Center crop the image
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(pil_image)
    
    # Normalize the image
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax