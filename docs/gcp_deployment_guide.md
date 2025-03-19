# Flowers Classification App: GCP Deployment Guide

This guide provides step-by-step instructions for deploying the Flowers Classification App on Google Cloud Platform (GCP). We will cover three deployment scenarios:

1. Training the model on a GCP virtual machine (optimized for CPU)
2. Deploying the prediction API for external access
3. Containerizing the entire application (including MLflow) on GCP

## Prerequisites

Before starting, ensure you have:

- A Google Cloud Platform account ([sign up here](https://cloud.google.com/free) if needed)
- Basic familiarity with GCP and its console
- The [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed (optional, as we'll be using Cloud Shell)
- Access to the Flowers Classification App repository

## Getting Started with Google Cloud Shell

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the Cloud Shell icon (>_) in the top right corner of the console
3. Once Cloud Shell is ready, click on the "Open Editor" button to launch the Cloud Shell Editor
4. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/flowers_classification_app.git
   cd flowers_classification_app
   ```

## 1. Setting Up a VM for Model Training (CPU Optimized)

### Step 1: Create a Cloud Storage Bucket for Data

```bash
# Set a unique bucket name
BUCKET_NAME="flowers-classification-data-$(date +%s)"

# Create the bucket
gsutil mb -l us-central1 gs://$BUCKET_NAME

# Export as an environment variable for later use
export BUCKET_NAME=$BUCKET_NAME
echo "Your bucket name is: $BUCKET_NAME"
```

### Step 2: Upload Training Data to the Bucket

If you already have the dataset locally:

```bash
# Create necessary folders in the bucket
gsutil mb -p gs://$BUCKET_NAME/data
gsutil mb -p gs://$BUCKET_NAME/data/raw
gsutil mb -p gs://$BUCKET_NAME/data/processed

# Upload the dataset
gsutil -m cp -r data/* gs://$BUCKET_NAME/data/
```

Alternatively, you can configure the app to download the dataset directly:

```bash
# Create a file to store the download URL
echo "https://drive.google.com/uc?export=download&id=18I2XurHF94K072w4rM3uwVjwFpP_7Dnz" > data_url.txt
gsutil cp data_url.txt gs://$BUCKET_NAME/data_url.txt
```

### Step 3: Modify the Code for GCP Integration

Create a new configuration file for GCP:

```bash
cat > configs/gcp_config.py << 'EOF'
import os

# GCP Configuration
GCP_BUCKET_NAME = os.environ.get('BUCKET_NAME', 'your-default-bucket-name')
GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')

# Storage paths
DATA_PATH = f"gs://{GCP_BUCKET_NAME}/data"
MODEL_PATH = f"gs://{GCP_BUCKET_NAME}/models"
OUTPUT_PATH = f"gs://{GCP_BUCKET_NAME}/outputs"

# MLflow settings
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Training settings
USE_GPU = False  # Set to False for CPU optimization
EOF
```

Update the data_processing.py file to work with GCS:

```bash
cat > src/utils/gcs_utils.py << 'EOF'
from google.cloud import storage
import os
import tempfile

def download_from_gcs(gcs_path, local_path):
    """Download a file from Google Cloud Storage to a local path."""
    # Parse bucket and blob path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
    
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    # Create the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Make sure the local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(local_path, gcs_path):
    """Upload a file from a local path to Google Cloud Storage."""
    # Parse bucket and blob path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
    
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    # Create the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload
    blob.upload_from_filename(local_path)
    return gcs_path

def list_gcs_files(gcs_path, extension=None):
    """List all files in a GCS path, optionally filtered by extension."""
    # Parse bucket and blob path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
    
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Create the client
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    # Filter by extension if provided
    if extension:
        return [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(extension)]
    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
EOF
```

### Step 4: Create a VM for Training

```bash
# Create a Compute Engine VM instance optimized for CPU-intensive tasks
gcloud compute instances create flowers-training-vm \
    --machine-type=n1-standard-4 \
    --boot-disk-size=50GB \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=flowers-app \
    --metadata=startup-script='#! /bin/bash
    # Install dependencies
    apt-get update
    apt-get install -y git python3-pip python3-venv

    # Clone the repository
    mkdir -p /opt/flowers
    cd /opt/flowers
    git clone https://github.com/your-repo/flowers_classification_app.git
    cd flowers_classification_app

    # Set up Python environment
    python3 -m venv .flower_classification
    source .flower_classification/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install google-cloud-storage

    # Install MLflow
    pip install mlflow==2.20.2
    
    # Set environment variables
    export BUCKET_NAME="'$BUCKET_NAME'"
    export PYTHONPATH=/opt/flowers/flowers_classification_app
    
    echo "Setup complete. Ready for training!"'
```

### Step 5: Connect to the VM and Run Training

```bash
# Connect to the VM
gcloud compute ssh flowers-training-vm

# Once connected to the VM, run:
cd /opt/flowers/flowers_classification_app
source .flower_classification/bin/activate

# Set environment variables
export BUCKET_NAME="your-bucket-name"  # Replace with your actual bucket name
export PYTHONPATH=/opt/flowers/flowers_classification_app

# Run the training script with CPU optimization
python src/train.py \
    --data_directory data/flowers \
    --arch vgg16 \
    --learning_rate 0.001 \
    --hidden_units 512 \
    --epochs 5 \
    --save_dir "gs://$BUCKET_NAME/models" \
    --save_name "model_checkpoint_cpu.pth" \
    --training_compute cpu \
    --freeze_parameters True
```

### Step 6: Monitor Training and Save Results

Training progress will be displayed in the terminal. Once complete:

```bash
# Verify the model was saved to the bucket
gsutil ls gs://$BUCKET_NAME/models/

# Copy MLflow artifacts to the bucket (if MLflow is used locally)
gsutil -m cp -r mlruns gs://$BUCKET_NAME/mlruns/
```

## 2. Deploying the Prediction API to Cloud Run

### Step 1: Create a Simplified Dockerfile for the API

```bash
cat > Dockerfile.api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for Cloud Run)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Google Cloud Storage
RUN pip install google-cloud-storage

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models predictions data/raw data/processed data/external

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.py

# Expose port for the API
EXPOSE 8080

# Default command runs the API on port 8080 (Cloud Run default)
CMD ["python", "-m", "gunicorn", "-b", ":8080", "predict_api:app"]
EOF
```

### Step 2: Update the predict_api.py to Work with GCS

Create a new file with modifications for GCP:

```bash
cat > gcp_predict_api.py << 'EOF'
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import logging
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision import models
from src.utils.image_normalization import process_image, imshow
from flask import Flask, jsonify, request
from src.utils.gcs_utils import download_from_gcs, upload_to_gcs, list_gcs_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Use environment variables with defaults for configuration
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'your-default-bucket-name')
CAT_NAMES_PATH = os.environ.get('CAT_NAMES_PATH', os.path.join(PROJECT_ROOT, "configs/cat_to_name.json"))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(PROJECT_ROOT, "models"))
GCS_MODELS_PATH = f"gs://{BUCKET_NAME}/models"

# Log the configuration
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Category names path: {CAT_NAMES_PATH}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"GCS models path: {GCS_MODELS_PATH}")

# Make sure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Download category names from GCS if needed
if not os.path.exists(CAT_NAMES_PATH):
    os.makedirs(os.path.dirname(CAT_NAMES_PATH), exist_ok=True)
    try:
        download_from_gcs(f"gs://{BUCKET_NAME}/configs/cat_to_name.json", CAT_NAMES_PATH)
    except Exception as e:
        logger.error(f"Error downloading cat_to_name.json: {e}")
        # Create a default category names file if download fails
        os.makedirs(os.path.dirname(CAT_NAMES_PATH), exist_ok=True)
        with open(CAT_NAMES_PATH, 'w') as f:
            f.write('{}')

with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

# Function definitions (process_base64_image, load_checkpoint, predict) remain the same as in predict_api.py

# Download models from GCS if not already present
def download_models_from_gcs():
    try:
        # List model files in GCS
        model_files = list_gcs_files(GCS_MODELS_PATH, extension=".pth")
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            local_path = os.path.join(MODELS_DIR, filename)
            
            if not os.path.exists(local_path):
                logger.info(f"Downloading model {filename} from GCS...")
                download_from_gcs(model_file, local_path)
                logger.info(f"Model {filename} downloaded successfully")
            else:
                logger.info(f"Model {filename} already exists locally")
                
        return True
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

# Try to download models at startup
download_models_from_gcs()

# (Include the rest of the predict_api.py functions and routes)

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    # Get host and port from environment variables (required for Cloud Run)
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port, debug=False)
EOF
```

### Step 3: Build and Deploy to Cloud Run

```bash
# Set project ID
PROJECT_ID=$(gcloud config get-value project)
echo $PROJECT_ID

# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/flowers-api --dockerfile Dockerfile.api

# Deploy to Cloud Run
gcloud run deploy flowers-api \
    --image gcr.io/$PROJECT_ID/flowers-api \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 1 \
    --allow-unauthenticated \
    --set-env-vars="BUCKET_NAME=$BUCKET_NAME"
```

### Step 4: Test the Deployed API

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe flowers-api --platform managed --region us-central1 --format 'value(status.url)')
echo $SERVICE_URL

# Test the API health endpoint
curl $SERVICE_URL/health

# Test a prediction (replace with your actual image path)
python - << END
import requests
import base64
import json

# Replace with your local image path
image_path = "path/to/your/test/image.jpg"

# Read and encode the image
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare request data
data = {
    "image_data": encoded_image,
    "top_k": 5
}

# Send request (replace with your actual service URL)
response = requests.post("$SERVICE_URL/v1/predict", json=data)

# Print results
print(json.dumps(response.json(), indent=2))
END
```

## 3. Deploying the Full Application with Containers on GKE

### Step 1: Set Up a Google Kubernetes Engine (GKE) Cluster

```bash
# Create a GKE cluster
gcloud container clusters create flowers-cluster \
    --num-nodes=2 \
    --machine-type=e2-standard-2 \
    --region=us-central1 \
    --project=$PROJECT_ID

# Get credentials for kubectl
gcloud container clusters get-credentials flowers-cluster \
    --region=us-central1 \
    --project=$PROJECT_ID
```

### Step 2: Update docker-compose.yml for GKE Deployment

Create a kubernetes configuration directory:

```bash
mkdir -p k8s
```

Create a persistent volume claim for MLflow:

```bash
cat > k8s/mlflow-pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
```

Create the MLflow deployment:

```bash
cat > k8s/mlflow-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: gcr.io/PROJECT_ID/flowers-mlflow:latest
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: mlflow-storage
          mountPath: /mlflow
        env:
        - name: BUCKET_NAME
          value: "BUCKET_NAME_PLACEHOLDER"
      volumes:
      - name: mlflow-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc
EOF
```

Create the MLflow service:

```bash
cat > k8s/mlflow-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: mlflow
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
EOF
```

Create the API deployment:

```bash
cat > k8s/api-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flowers-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flowers-api
  template:
    metadata:
      labels:
        app: flowers-api
    spec:
      containers:
      - name: flowers-api
        image: gcr.io/PROJECT_ID/flowers-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        - name: BUCKET_NAME
          value: "BUCKET_NAME_PLACEHOLDER"
EOF
```

Create the API service:

```bash
cat > k8s/api-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: flowers-api
spec:
  selector:
    app: flowers-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
EOF
```

### Step 3: Build and Push the Container Images

```bash
# Update the Kubernetes YAML files with actual values
sed -i "s/PROJECT_ID/$PROJECT_ID/g" k8s/*.yaml
sed -i "s/BUCKET_NAME_PLACEHOLDER/$BUCKET_NAME/g" k8s/*.yaml

# Build and push the API container
gcloud builds submit --tag gcr.io/$PROJECT_ID/flowers-api .

# Build and push the MLflow container
gcloud builds submit --tag gcr.io/$PROJECT_ID/flowers-mlflow --dockerfile Dockerfile.mlflow .
```

### Step 4: Deploy to GKE

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# Check deployment status
kubectl get deployments
kubectl get services
```

### Step 5: Access the Services

```bash
# Get the API service external IP
API_IP=$(kubectl get service flowers-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Flowers API is available at: http://$API_IP/flowers_classification_home"

# For MLflow (internal to the cluster), we need to set up port forwarding to access it
kubectl port-forward service/mlflow 5000:5000
# Then access MLflow at http://localhost:5000
```

## Integration Tips and Best Practices

### Data Management

1. **Use GCS efficiently**:
   ```bash
   # Create lifecycle rules to automatically delete older artifacts
   gsutil lifecycle set lifecycle-config.json gs://$BUCKET_NAME
   ```

2. **Consider versioning for the data**:
   ```bash
   # Enable versioning on the bucket
   gsutil versioning set on gs://$BUCKET_NAME
   ```

3. **Set appropriate IAM permissions**:
   ```bash
   # Grant storage access to the default compute service account
   gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
       --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
       --role=roles/storage.objectAdmin
   ```

### Cost Optimization

1. Use preemptible VMs for training to reduce costs by 70-80%
2. Set up automatic shutdown of VMs after training completes
3. Use Cloud Run for the API, which only charges for actual usage
4. Configure autoscaling for GKE to scale down during periods of low activity

### Monitoring and Logging

1. Set up Cloud Monitoring dashboards for your services
2. Configure alerts for unusual API traffic or errors
3. Use Cloud Logging to centralize logs from all components

### Security Best Practices

1. Use service accounts with minimal permissions
2. Store sensitive configuration in Secret Manager
3. Configure VPC Service Controls to restrict access to your resources
4. Regularly update container images to patch security vulnerabilities

## Conclusion

This guide provides a comprehensive approach to deploying the Flowers Classification App on GCP. By following these instructions, you can:

1. Train models efficiently on CPU-optimized VMs
2. Deploy a prediction API that's accessible to external users
3. Set up a containerized environment with MLflow for tracking experiments

Remember to clean up resources when they're no longer needed to avoid unnecessary charges. Consider implementing CI/CD pipelines to automate deployments as your project evolves.
