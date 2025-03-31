# Explainable AI (XAI) in Flowers Classification

## Table of Contents
1. [Introduction to Explainable AI](#introduction-to-explainable-ai)
2. [Key XAI Techniques](#key-xai-techniques)
   - [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
   - [LIME (Local Interpretable Model-agnostic Explanations)](#lime-local-interpretable-model-agnostic-explanations)
3. [Implementation in the Prediction API](#implementation-in-the-prediction-api)
   - [Architecture Overview](#architecture-overview)
   - [Modular Structure](#modular-structure)
   - [XAI Pipeline](#xai-pipeline)
4. [Using the XAI Endpoints](#using-the-xai-endpoints)
   - [Example Setup](#example-setup)
   - [Prediction Endpoints](#prediction-endpoints)
   - [SHAP Explanations](#shap-explanations)
   - [LIME Explanations](#lime-explanations)
5. [Understanding the Visualizations](#understanding-the-visualizations)
6. [Global vs. Local Explainability](#global-vs-local-explainability)
7. [Best Practices and Considerations](#best-practices-and-considerations)

## Introduction to Explainable AI

Explainable AI (XAI) refers to methods and techniques that make the behavior and predictions of artificial intelligence systems understandable to humans. As machine learning models become more complex (especially deep learning models like the flower classifiers in this application), understanding why they make specific decisions becomes increasingly difficult.

XAI aims to address the "black box" problem of complex models by:

- **Transparency**: Providing insights into how models process inputs to generate outputs
- **Interpretability**: Making model decisions comprehensible to humans
- **Accountability**: Enabling verification that models are making decisions based on relevant features
- **Trust**: Building confidence in model predictions by explaining their reasoning

In the context of flower classification, XAI helps us understand which parts of an image contribute most to identifying a specific flower type. This information is valuable for:

- **Debugging Models**: Identifying when models are using irrelevant background features
- **Building Trust**: Showing users why a particular classification was made
- **Educational Purposes**: Teaching students how neural networks "see" and classify images
- **Model Improvement**: Identifying confusing patterns or error sources

## Key XAI Techniques

Our implementation focuses on two powerful XAI techniques:

### SHAP (SHapley Additive exPlanations)

SHAP is based on cooperative game theory and provides a unified measure of feature importance. Our implementation includes:

- **GradientSHAP**: An efficient approach that combines ideas from gradient-based methods with the SHAP framework. It extracts feature importance directly using gradients and works well with deep learning models.

- **DeepSHAP**: This method approximates SHAP values for deep learning models. It propagates activation differences through the network from a reference input to the current prediction.

- **Custom SHAP fallback**: When the standard SHAP methods are too computationally intensive or fail, our implementation falls back to a simpler occlusion-based approach via the `generate_custom_shap_explanation` function in `src/explanations/fallback.py` that produces similar explanations with better performance.

SHAP values show how each feature (in our case, each pixel) contributes positively or negatively to the predicted class. The visualizations use a blue-red colormap, where red areas indicate positive contribution to the predicted class and blue areas indicate negative contribution.

### LIME (Local Interpretable Model-agnostic Explanations)

LIME works by creating interpretable surrogate models that approximate the behavior of the complex model around specific instances. For images, LIME:

1. Divides the image into superpixels (segments of similar pixels)
2. Creates variations of the input by hiding random subsets of these superpixels 
3. Observes how the model's predictions change with these variations
4. Builds a simple linear model that approximates the complex model's behavior

Our implementation uses the official `lime_image` package with optimized parameters for flower images. The visualization shows which segments (superpixels) of the image most contributed to the prediction.

## Implementation in the Prediction API

### Architecture Overview

The Flowers Classification API provides both standard prediction endpoints and explainable AI endpoints that help users understand why predictions were made. The API is built with Flask and is designed with modularity and maintainability in mind.

### Modular Structure

The application has been restructured into a modular architecture to improve maintainability and separation of concerns:

```
flowers_classification_app/
├── api_main.py                  # API entry point
├── src/
│   ├── predict_api.py           # Main Flask application and routes
│   ├── explanations/
│   │   ├── base.py              # Core visualization functionality
│   │   ├── shap_explanations.py # SHAP explanation methods
│   │   ├── lime_explanations.py # LIME explanation methods
│   │   └── fallback.py          # Fallback explanation methods
│   ├── utils/
│   │   ├── image_processing.py  # Image preprocessing
│   │   └── model_utils.py       # Model loading and prediction
│   └── metrics/
│       └── prometheus_metrics.py # Monitoring utilities
```

This structure separates the core functionality into focused modules, making the codebase more maintainable and easier to extend.

### XAI Pipeline

The explainable AI pipeline follows these steps:

1. **Image Preprocessing**: The base64-encoded image is processed and normalized
2. **Prediction**: The model makes a prediction on the processed image
3. **Explanation Generation**: Either SHAP or LIME explanations are generated based on the endpoint used
4. **Visualization**: A standardized visualization is created showing the original image, feature importance, and an overlay
5. **Response**: The API returns a JSON response containing URLs to the visualization and prediction details

The explanation pipeline includes automatic fallback mechanisms if the primary explanation method fails, ensuring robustness in production environments.

## Using the XAI Endpoints

### Example Setup

The API can be run using the `api_main.py` script:

```bash
python api_main.py
```

This will start the Flask server on the default host (`0.0.0.0`) and port (`9000`), which can be customized via environment variables:

```bash
FLASK_HOST=127.0.0.1 FLASK_PORT=8080 python api_main.py
```

### Prediction Endpoints

The API provides two versions of prediction endpoints:

#### Example 1: Basic Prediction (v1)

Simple Python script to make a prediction request:

```python
import os
import base64
import json
import requests

# Load and encode the image
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make the prediction request
response = requests.post(
    'http://localhost:9000/v1/predict',
    json={'image_data': image_data}
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

You can run this directly as a one-liner:

```bash
python -c "
import os, base64, json, requests
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
response = requests.post('http://localhost:9000/v1/predict', json={'image_data': image_data})
print(json.dumps(response.json(), indent=2))
"
```

#### Example 2: Enhanced Prediction (v2) with Top-K Results

```python
import os
import base64
import json
import requests

# Load and encode the image
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make the prediction request with top_k parameter
response = requests.post(
    'http://localhost:9000/v2/predict',
    json={'image_data': image_data, 'top_k': 5, 'model_version': 'v2'}
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

One-liner version:

```bash
python -c "
import os, base64, json, requests
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
response = requests.post('http://localhost:9000/v2/predict', json={'image_data': image_data, 'top_k': 5, 'model_version': 'v2'})
print(json.dumps(response.json(), indent=2))
"
```

### SHAP Explanations

The API provides SHAP explanation endpoints that generate visual explanations of model predictions.

To request a SHAP explanation:

```
POST /v1/explain_shap
Content-Type: application/json

{
  "image_data": "base64_encoded_image_string",
  "top_k": 5
}
```

Example response:
```json
{
  "success": true,
  "explanation": {
    "visualization": "base64_encoded_explanation_image",
    "prediction": {
      "probabilities": [0.9, 0.05, 0.03, 0.01, 0.01],
      "classes": ["21", "63", "12", "46", "57"],
      "class_names": ["Fire Lily", "Black-Eyed Susan", "Colt's Foot", "Fritillary", "Hard-Leaved Pocket Orchid"]
    },
    "explanation_type": "SHAP",
    "description": "SHAP values show the contribution of each pixel to the prediction."
  }
}
```

#### Example 3: SHAP Explanation Python Script

```python
import os
import base64
import json
import requests

# Load and encode the image
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Request SHAP explanation
response = requests.post(
    'http://localhost:9000/v2/explain_shap',
    json={'image_data': image_data, 'top_k': 5}
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

Running as a one-liner:

```bash
python -c "
import os, base64, json, requests
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
response = requests.post('http://localhost:9000/v2/explain_shap', json={'image_data': image_data, 'top_k': 5})
print(json.dumps(response.json(), indent=2))
"
```

Using Flask's test client (useful for integration tests):

```python
import os
import base64
import json
from src.predict_api import app

app.testing = True
client = app.test_client()
image_path = os.path.join('tests', 'image_08115.jpg')

with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.post('/v2/explain_shap', json={'image_data': image_data, 'top_k': 5})
print(json.dumps(response.json, indent=2))
```

### LIME Explanations

The API also provides LIME explanation endpoints that use a different approach to explain predictions.

To request a LIME explanation:

```
POST /v1/explain_lime
Content-Type: application/json

{
  "image_data": "base64_encoded_image_string",
  "top_k": 5,
  "num_samples": 500
}
```

Example response:
```json
{
  "success": true,
  "explanation": {
    "visualization": "base64_encoded_explanation_image",
    "prediction": {
      "probabilities": [0.9, 0.05, 0.03, 0.01, 0.01],
      "classes": ["21", "63", "12", "46", "57"],
      "class_names": ["Fire Lily", "Black-Eyed Susan", "Colt's Foot", "Fritillary", "Hard-Leaved Pocket Orchid"]
    },
    "explanation_type": "LIME",
    "description": "LIME identifies regions of the image that contribute most to the prediction."
  }
}
```

#### Example 4: LIME Explanation Python Script

```python
import os
import base64
import json
import requests

# Load and encode the image
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Request LIME explanation
response = requests.post(
    'http://localhost:9000/v2/explain_lime',
    json={'image_data': image_data, 'top_k': 5, 'num_samples': 500}
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

Running as a one-liner:

```bash
python -c "
import os, base64, json, requests
image_path = os.path.join('tests', 'image_08115.jpg')
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
response = requests.post('http://localhost:9000/v2/explain_lime', json={'image_data': image_data, 'top_k': 5, 'num_samples': 500})
print(json.dumps(response.json(), indent=2))
"
```

Using Flask's test client:

```python
import os
import base64
import json
from src.predict_api import app

app.testing = True
client = app.test_client()
image_path = os.path.join('tests', 'image_08115.jpg')

with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.post('/v2/explain_lime', json={'image_data': image_data, 'top_k': 5, 'num_samples': 500})
print(json.dumps(response.json, indent=2))
```

## Understanding the Visualizations

The explanation endpoints return URLs to visualization images that include three panels:

1. **Original Image**: The unmodified input image
2. **Feature Importance**: A heatmap showing which areas of the image influenced the prediction most strongly
3. **Importance Overlay**: A combination of the original image and the importance heatmap, showing how the features relate to the original image

### SHAP Visualizations

SHAP visualizations provide a heatmap that indicates:
- **Red areas**: Pixels that increase the probability of the predicted class
- **Blue areas**: Pixels that decrease the probability of the predicted class
- **Intensity**: The magnitude of the pixel's influence on the prediction

A good SHAP explanation for a flower classifier would typically show strong red highlights on the distinctive parts of the flower (petals, stamen, unique patterns) and blue or neutral coloring on irrelevant background elements.

In our implementation, we use `DeepExplainer` from the SHAP library, which is specifically designed for deep learning models. It creates a background distribution of inputs and calculates how each pixel influences the final prediction relative to this background.

Key concepts in SHAP:
- **Attribution**: SHAP assigns each feature (in our case, pixels or regions in the image) an importance value for a particular prediction
- **Consistency**: Features contributing similarly across different instances receive similar importance scores
- **Global & Local**: SHAP can explain both individual predictions (local) and overall model behavior (global)

### LIME Visualizations

LIME creates explanations by approximating the complex model with a simpler, interpretable model around a specific prediction.

**How LIME Works:**

1. **Perturbation**: Generate perturbed versions of the input by manipulating parts of the image
2. **Prediction**: Obtain predictions from the complex model for these perturbed inputs
3. **Simplified Model**: Train a simpler model (like linear regression) on the perturbed dataset
4. **Interpretation**: Identify which regions have the most influence on the prediction

In our implementation, we use the `LimeImageExplainer` which partitions the image into interpretable components (superpixels) and measures how masking different regions affects the prediction.

## Implementation in the Prediction API

### Architecture Overview

The Flowers Classification API has been enhanced to include XAI capabilities. The existing architecture follows:

- **Base API**: A Flask-based REST API serving two versions of the flower classification model
- **Docker Container**: Runs on port 9000 as described in the Docker setup
- **Prediction Endpoints**: `/v1/predict` and `/v2/predict` for different model versions

The new XAI features add four additional endpoints:
- `/v1/explain_shap`: SHAP explanations for model v1
- `/v2/explain_shap`: SHAP explanations for model v2
- `/v1/explain_lime`: LIME explanations for model v1
- `/v2/explain_lime`: LIME explanations for model v2

### XAI Pipeline

The implementation consists of several key components:

1. **Model Wrapper**: Enhances the model interface for compatibility with XAI tools
   ```python
   class ModelWrapper(nn.Module):
       def __init__(self, model):
           super(ModelWrapper, self).__init__()
           self.model = model
           self.device = next(model.parameters()).device
           
       def forward(self, x):
           # Ensure input is on the correct device
           x = x.to(self.device)
           return self.model(x)
   ```

2. **SHAP Explanation Generator**:
   ```python
   def generate_shap_explanation(image_data, model, top_k=5):
       # Create background dataset
       background = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
       
       # Initialize SHAP explainer
       explainer = shap.DeepExplainer(model_wrapper, background)
       
       # Compute SHAP values
       shap_values = explainer.shap_values(tensor_image)
       
       # Generate visualization
       shap.image_plot(shap_values, image_for_plot)
   ```

3. **LIME Explanation Generator**:
   ```python
   def generate_lime_explanation(image_data, model, top_k=5):
       # Initialize LIME explainer
       explainer = lime_image.LimeImageExplainer()
       
       # Get explanation
       explanation = explainer.explain_instance(
           image_for_lime, 
           predict_fn,
           top_labels=top_k,
           hide_color=0,
           num_samples=1000
       )
       
       # Generate visualization
       temp, mask = explanation.get_image_and_mask(top_class)
       plt.imshow(mark_boundaries(temp, mask))
   ```

4. **API Endpoints**: Handle requests, process images, generate explanations, and return results

Response:
```json
{
  "success": true,
  "explanation": {
    "visualization": "base64_encoded_explanation_image",
    "prediction": {
      "probabilities": [0.9, 0.05, 0.03, 0.01, 0.01],
      "classes": ["21", "63", "12", "46", "57"],
      "class_names": ["Fire Lily", "Black-Eyed Susan", "Colt's Foot", "Fritillary", "Hard-Leaved Pocket Orchid"]
    },
    "explanation_type": "SHAP",
    "description": "SHAP values show the contribution of each pixel to the prediction."
  }
}
```

### LIME Explanations

To request a LIME explanation:

```
POST /v1/explain_lime
Content-Type: application/json

{
  "image_data": "base64_encoded_image_string",
  "top_k": 5
}
```

Response:
```json
{
  "success": true,
  "explanation": {
    "visualization": "base64_encoded_explanation_image",
    "prediction": {
      "probabilities": [0.9, 0.05, 0.03, 0.01, 0.01],
      "classes": ["21", "63", "12", "46", "57"],
      "class_names": ["Fire Lily", "Black-Eyed Susan", "Colt's Foot", "Fritillary", "Hard-Leaved Pocket Orchid"]
    },
    "explanation_type": "LIME",
    "description": "LIME identifies regions of the image that contribute most to the prediction."
  }
}
```

## Global vs. Local Explainability

Our implementation provides both perspectives on model behavior:

**Local Explainability** (implemented directly):
- Explains individual predictions
- Shows what influenced a specific classification decision
- Useful for understanding particular instances

**Global Explainability** (can be derived):
- By aggregating multiple SHAP explanations, we can identify patterns of important features across the dataset
- Helps understand overall model behavior

## Best Practices and Considerations

When using the XAI features:

1. **Computational Overhead**: XAI techniques, especially SHAP, can be computationally intensive. Consider this in production environments.

2. **Interpretation Context**: Always interpret explanations with domain knowledge about flowers and their identifying characteristics.

3. **Complementary Techniques**: Use both SHAP and LIME for a more comprehensive understanding:
   - SHAP for pixel-level contributions
   - LIME for higher-level region importance

4. **Background Selection**: The SHAP implementation uses a simple background of zeros, but more representative backgrounds could potentially provide better explanations.

5. **Visualization Interpretation**: Remember that these visualizations are approximations and simplifications of complex model behavior.

6. **Model Improvement**: Use explanations to identify when the model is focusing on irrelevant features and refine your training accordingly.

7. **Educational Value**: These visualizations are particularly valuable for teaching AI concepts and building intuition about neural network behavior.

By leveraging these XAI techniques, we can make our flower classification models more transparent, trustworthy, and instructive, enabling users to understand not just what the model predicts, but why it makes those predictions.
