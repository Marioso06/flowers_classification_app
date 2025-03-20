# Flowers Classification App: GCP Deployment Guide

This guide provides step-by-step instructions for deploying the Flowers Classification App on Google Cloud Platform (GCP). The deployment workflow follows these logical steps:

1. Setting up cloud storage and uploading data
2. Deploying MLflow with Cloud SQL and GCS integration
3. Setting up a VM for model training
4. Training models and tracking experiments with MLflow
5. Deploying the prediction API to Cloud Run

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

## 1. Setting Up Cloud Storage and Uploading Data

### Step 1: Create a Cloud Storage Bucket

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

First we would need to get the data into our cloud environment by executing the following commands:

```bash
mkdir -p data/raw && cd data/raw
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=18I2XurHF94K072w4rM3uwVjwFpP_7Dnz&export=download&confirm=t&uuid=7feba52a-4578-499a-b3bc-469e687781f4" -O flower_data.tar.gz
tar -xzf flower_data.tar.gz
cd ../..

# Upload data to the bucket
gsutil -m cp -r data/raw/flower_data gs://$BUCKET_NAME/data/

# Verify data was uploaded
gsutil ls -r gs://$BUCKET_NAME/data/ | head -10
```

### Step 3: Verify the GCP Configuration

The app already comes with a GCP configuration file that contains settings for bucket names, MLflow, and database configuration. Let's verify it:

```bash
# View the GCP configuration file content
cat configs/gcp_config.py
```

Verify that the GCS utility functions are available in `src/utils/gcs_utils.py`:

```bash
# Check that GCS utilities exist
cat src/utils/gcs_utils.py
```

These utilities allow the application to download/upload files from/to Google Cloud Storage.

## 2. Setting Up MLflow with Cloud SQL and GCS

### Step 1: Create a Cloud SQL PostgreSQL Instance

```bash
# Create a PostgreSQL instance
gcloud sql instances create mlflow-db \
    --database-version=POSTGRES_13 \
    --cpu=1 \
    --memory=3840MB \
    --region=us-central1 \
    --root-password="SECURE_PASSWORD_HERE" \
    --storage-size=10GB \
    --storage-type=SSD

# Create a database for MLflow
gcloud sql databases create mlflow \
    --instance=mlflow-db

# Create a user for MLflow
gcloud sql users create mlflow \
    --instance=mlflow-db \
    --password="MLFLOW_USER_PASSWORD_HERE"

# Get the connection details
GCLOUD_DB_IP=$(gcloud sql instances describe mlflow-db --format='value(ipAddresses[0].ipAddress)')
echo "Cloud SQL IP: $GCLOUD_DB_IP"

# Store these securely
export DB_HOST=$GCLOUD_DB_IP
export DB_USER="mlflow"
export DB_PASSWORD="MLFLOW_USER_PASSWORD_HERE"
export DB_NAME="mlflow"
```

### Step 2: Create a Compute Engine VM for MLflow (Installation Only)

```bash
gcloud compute instances create mlflow-server \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --boot-disk-size=20GB \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=mlflow-server \
    --metadata=startup-script='#!/bin/bash
exec > /var/log/startup-script.log 2>&1
set -x

# Update and install dependencies
apt-get update
apt-get install -y git python3-pip python3-venv postgresql-client

# Clone the repository
mkdir -p /opt/mlflow
cd /opt/mlflow
git clone https://github.com/Marioso06/flowers_classification_app.git
cd flowers_classification_app
git checkout lab_cloud_gcp

# Set up Python environment
python3 -m venv .mlflow_env
source .mlflow_env/bin/activate
pip install --upgrade pip
pip install mlflow==2.20.2 google-cloud-storage psycopg2-binary

# Create a configuration file for running MLflow later
cat > /opt/mlflow/run_mlflow.sh << EOF
#!/bin/bash
cd /opt/mlflow/flowers_classification_app
source .mlflow_env/bin/activate

# Set environment variables
export BUCKET_NAME="$BUCKET_NAME"
export USE_GCS_FOR_MLFLOW="true"
export MLFLOW_HOST="0.0.0.0"
export MLFLOW_PORT="5000"
export DB_HOST="$DB_HOST"
export DB_USER="$DB_USER"
export DB_PASSWORD="$DB_PASSWORD"
export DB_NAME="$DB_NAME"
export MLFLOW_DB_URI="postgresql://\$DB_USER:\$DB_PASSWORD@\$DB_HOST:5432/\$DB_NAME"
export USE_POSTGRES_FOR_MLFLOW="true"
export PYTHONPATH=/opt/mlflow/flowers_classification_app:\$PYTHONPATH

# Start MLflow
nohup python src/utils/mlflow_initialization.py --host 0.0.0.0 --port 5000 --use-gcs --use-postgres > mlflow.log 2>&1 &
echo "MLflow server started on port 5000"
EOF

chmod +x /opt/mlflow/run_mlflow.sh
echo "VM setup complete. Ready to start MLflow after Cloud SQL authorization."
'
```

### Step 3: Allow MLflow Traffic

```bash
# Create a firewall rule to allow traffic to the MLflow server
gcloud compute firewall-rules create allow-mlflow \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:5000 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=mlflow-server
```

### Step 4: Authorize the VM to Connect to Cloud SQL

By default, Cloud SQL instance connections are blocked. You need to authorize your VM's IP address to connect to your PostgreSQL instance:

```bash
# Get the VM's external IP
VM_IP=$(gcloud compute instances describe mlflow-server --zone=us-central1-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "VM IP: $VM_IP"

# Add the VM's IP to Cloud SQL authorized networks
gcloud sql instances patch mlflow-db --authorized-networks=$VM_IP
```

### Step 5: Start MLflow Server

Now that the network is properly configured, start the MLflow server:

```bash
# SSH into the MLflow VM and run the prepared script
gcloud compute ssh mlflow-server --zone=us-central1-a --command="sudo /opt/mlflow/run_mlflow.sh"

# Verify the connection to the database
gcloud compute ssh mlflow-server --zone=us-central1-a --command="PGPASSWORD='$DB_PASSWORD' psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c '\conninfo'"

# Verify MLflow is running
gcloud compute ssh mlflow-server --zone=us-central1-a --command="ps aux | grep mlflow && netstat -tuln | grep 5000"
```

If you encounter issues:

```bash
# SSH into the MLflow VM for troubleshooting
gcloud compute ssh mlflow-server --zone=us-central1-a

# Check logs
sudo cat /var/log/startup-script.log
cat /opt/mlflow/flowers_classification_app/mlflow.log

# Manually start MLflow if needed
sudo /opt/mlflow/run_mlflow.sh
```

### Step 6: Get the MLflow Server External IP

```bash
MLFLOW_IP=$(gcloud compute instances describe mlflow-server --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "MLflow server is available at: http://$MLFLOW_IP:5000"

# Export this for use in other commands
export MLFLOW_TRACKING_URI="http://$MLFLOW_IP:5000"
```

### Step 7: Set Up Environment Security (Optional but Recommended)

For production, store sensitive connection information in Secret Manager:

```bash
# Create secrets for database credentials
gcloud secrets create mlflow-db-uri \
  --replication-policy="automatic" \
  --data-file=<(echo "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:5432/$DB_NAME")

# Grant access to the MLflow VM service account
MLFLOW_VM_SA=$(gcloud compute instances describe mlflow-server --format='value(serviceAccounts.email)')
gcloud secrets add-iam-policy-binding mlflow-db-uri \
  --member="serviceAccount:$MLFLOW_VM_SA" \
  --role="roles/secretmanager.secretAccessor"
```

Then update your VM startup script to use secrets:

```bash
# In the VM startup script
export MLFLOW_BACKEND_STORE_URI=$(gcloud secrets versions access latest --secret="mlflow-db-uri")
```

## 3. Setting Up a VM for Model Training

### Step 1: Create a Training VM

```bash
# Create a Compute Engine VM instance optimized for CPU-intensive tasks
gcloud compute instances create flowers-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --boot-disk-size=200GB \
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
    git clone https://github.com/Marioso06/flowers_classification_app.git
    mkdir -p /opt/flowers/flowers_classification_app/data_temp
    chown -R $USER:$USER /opt/flowers/flowers_classification_app/data_temp
    chmod -R 755 /opt/flowers/flowers_classification_app/data_temp
    cd flowers_classification_app
    git checkout lab_cloud_gcp
    # Set up Python environment
    make init-cpu
    
    echo "Setup complete. Ready for training!"'
```

## 4. Training Models and Tracking Experiments with MLflow

### Step 1: Connect to the Training VM

```bash
# Connect to the VM
gcloud compute ssh flowers-training-vm

# Once connected to the VM, run:
cd /opt/flowers/flowers_classification_app
source .flower_classification/bin/activate
```

### Step 2: Configure Environment and Run Training

```bash
# Set environment variables
export BUCKET_NAME="your-bucket-name"  # Replace with your actual bucket name
export PYTHONPATH=/opt/flowers/flowers_classification_app

# Configure MLflow to use the remote tracking server
export MLFLOW_TRACKING_URI="http://$MLFLOW_IP:5000"  # Replace with your MLflow server IP

# Run the training script with CPU optimization
python src/train.py \
    --data_directory data/flowers \
    --arch vgg13 \
    --learning_rate 0.001 \
    --hidden_units 512 \
    --epochs 1 \
    --save_dir "gs://$BUCKET_NAME/models" \
    --save_name "model_checkpoint_cpu.pth" \
    --training_compute cpu \
    --freeze_parameters True \
    --bucket_name "$BUCKET_NAME"
```

### Step 3: Track Experiments with MLflow

The application is configured to automatically track experiments with MLflow. You can access the MLflow UI by visiting the MLflow server URL:

```
http://$MLFLOW_IP:5000
```

Alternatively, you can port-forward from your local machine if needed:

```bash
# From your local machine, set up port forwarding to the MLflow VM
gcloud compute ssh mlflow-server -- -L 5000:localhost:5000

# Then open a browser to http://localhost:5000
```

### Step 4: Save and Verify Results

```bash
# Verify that models were saved to the bucket
gsutil ls gs://$BUCKET_NAME/models/

# Verify MLflow artifacts
gsutil ls gs://$BUCKET_NAME/mlflow-artifacts/
```

## 5. Deploying the Prediction API to Cloud Run

### Step 1: Create a Simplified Dockerfile for the API

```bash
cat > Dockerfile.api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for Cloud Run)
RUN pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu

# Install Google Cloud Storage and MLflow
RUN pip install google-cloud-storage mlflow==2.20.2

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models predictions data/raw data/processed data/external

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.py
ENV USE_GCS_FOR_MLFLOW=true

# Expose port for the API
EXPOSE 8080

# Default command runs the API on port 8080 (Cloud Run default)
CMD ["python", "-m", "gunicorn", "-b", ":8080", "predict_api:app"]
EOF
```

### Step 2: Build and Deploy to Cloud Run

```bash
# Set environment variables
export BUCKET_NAME="your-bucket-name"
export PROJECT_ID="your-gcp-project-id"
export MLFLOW_IP=$(gcloud compute instances describe mlflow-server --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Build the container image
docker build -t gcr.io/$PROJECT_ID/flowers-api:latest -f Dockerfile.api .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/flowers-api:latest

# Deploy to Cloud Run with MLflow configuration
gcloud run deploy flowers-api \
  --image gcr.io/$PROJECT_ID/flowers-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars="BUCKET_NAME=$BUCKET_NAME,USE_GCS_FOR_MLFLOW=true,MLFLOW_TRACKING_URI=http://$MLFLOW_IP:5000"

# If you want to use MLflow for model loading, add these env vars
# --set-env-vars="USE_MLFLOW_MODELS=true,MLFLOW_RUN_ID_V1=your-run-id,MLFLOW_RUN_ID_V2=your-run-id"
```

### Step 3: Test the Deployed API

```bash
# Set project ID
PROJECT_ID=$(gcloud config get-value project)
echo $PROJECT_ID

# Get the URL of your deployed service
SERVICE_URL=$(gcloud run services describe flowers-api --platform managed --region us-central1 --format="value(status.url)")
echo $SERVICE_URL

# Test the health endpoint
curl -X GET $SERVICE_URL/health

# Test a prediction with a sample image (replace with your image URL)
curl -X POST $SERVICE_URL/predict/v1 \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://storage.googleapis.com/flower-sample-images/daisy.jpg"}'
```

## 6. Monitoring and Maintenance

### Setting Up Monitoring for the MLflow Server

```bash
# Create a simple uptime check for the MLflow server
gcloud monitoring uptime-check-configs create mlflow-server-check \
  --display-name="MLflow Server Uptime Check" \
  --resource-type=uptime-url \
  --http-check-path=/  \
  --http-check-port=5000 \
  --timeout=10s \
  --check-interval=5m \
  --content-matcher="MLflow" \
  --peer-project-id=$PROJECT_ID \
  --monitored-resource=instance \
  --host=$MLFLOW_IP
```

### Create Scheduled Backups for the Database

```bash
# Set up automatic backups for the Cloud SQL instance
gcloud sql instances patch mlflow-db \
  --backup-start-time="23:00" \
  --enable-bin-log \
  --retained-backups-count=7
```

## Conclusion

This guide provides a comprehensive approach to deploying the Flowers Classification App on GCP with a focus on production-ready MLflow integration. By following these steps, you have:

1. Set up cloud storage for your data and models
2. Deployed MLflow with a PostgreSQL backend for reliable experiment tracking
3. Created a training environment for your models
4. Deployed a scalable prediction API on Cloud Run
5. Configured monitoring and maintenance for your infrastructure

The application is now ready for production use with proper database backing, cloud storage integration, and scalable API endpoints.


 export BUCKET_NAME="flowers-classification-data-1-1742430057"
(.flower_classification) marioasmca@flowers-training-vm:/opt/flowers/flowers_classification_app$ export PYTHONPATH=/opt/flowers/flowers_classification_app
(.flower_classification) marioasmca@flowers-training-vm:/opt/flowers/flowers_classification_app$ export MLFLOW_TRACKING_URI="http://34.173.41.21:5000"