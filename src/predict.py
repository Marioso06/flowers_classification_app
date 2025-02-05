import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from torchvision import models
from utils.arg_parser import get_input_args
from utils.image_normalization import process_image, imshow

# Define the root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Create the predictions directory if it doesn't exist
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

class Predict:
    def __init__(self, category_names, top_k, checkpoint, image_path):
        self.category_names = category_names
        self.top_k = top_k
        self.checkpoint = checkpoint
        self.model = self.load_checkpoint(checkpoint)
        self.image_path = image_path

    def load_checkpoint(self, checkpoint_path):
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            # Load the model architecture
            architecture = checkpoint.get('architecture', 'vgg19')
            model = getattr(models, architecture)(pretrained=True)
            
            # Rebuild the classifier
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

    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model. '''
        
        # Process the image
        np_image = process_image(self.image_path)
        
        # Convert to PyTorch tensor
        tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        # Move model and tensor to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        tensor_image = tensor_image.to(device)
        
        # Set the model to evaluation mode and make predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_image)
        
        # Convert output to probabilities
        ps = torch.exp(output)
        
        # Get the top k probabilities and classes
        top_ps, top_indices = ps.topk(self.top_k, dim=1)
        
        # Move to CPU and convert to lists
        top_ps = top_ps.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()
        
        # Ensure class_to_idx exists
        if not hasattr(self.model, "class_to_idx"):
            raise AttributeError("Model does not have class_to_idx attribute.")
        
        # Invert class_to_idx mapping
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        
        # Map indices to actual class labels
        top_classes = [idx_to_class[idx] for idx in top_indices]
        
        return top_ps, top_classes


if __name__ == "__main__":
    print('===================== Prediction Started! =====================')
    
    # Parse arguments
    in_arg = get_input_args()
    
    # Load category mapping
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Run prediction
    predictor = Predict(
        cat_to_name,
        in_arg.top_k,
        in_arg.checkpoint,
        in_arg.image_path
    )
    
    probs, classes = predictor.predict()
    class_names = [cat_to_name[name] for name in classes]

    # Process image for plotting
    image = process_image(in_arg.image_path)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Show image
    imshow(image, ax=ax1, title=class_names[0])

    # Create probability bar chart
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Class Predictions')

    plt.tight_layout()

    # Generate a unique filename
    image_name = os.path.basename(in_arg.image_path)  # Extract filename from path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current time
    save_path = os.path.join(PREDICTIONS_DIR, f"prediction_{image_name}_{timestamp}.png")

    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent inline display

    print(f'✅ Prediction saved: {save_path}')
    print('===================== Prediction completed! =====================')