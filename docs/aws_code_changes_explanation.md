# AWS Code Changes Explanation

This document provides a comprehensive explanation of the code changes made to adapt the Flowers Classification App for deployment on Amazon Web Services (AWS). The project has been modified to leverage AWS services for cloud storage, database, and compute resources.

## 1. Configuration Changes

### AWS Configuration File (`configs/aws_config.py`)

A dedicated configuration file was created to centralize all AWS-related settings:

```python
import os

# AWS Configuration
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME', 'flowers-classification-aws')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')  # Set via environment variable
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')  # Set via environment variable

# Storage paths
DATA_PATH = f"s3://{AWS_S3_BUCKET_NAME}/data"
MODEL_PATH = f"s3://{AWS_S3_BUCKET_NAME}/models"
OUTPUT_PATH = f"s3://{AWS_S3_BUCKET_NAME}/outputs"

# MLflow settings
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_ARTIFACTS_PATH = os.environ.get('MLFLOW_ARTIFACTS_PATH', 'mlflow-artifacts')
MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'flowers-classification')

# Database settings - use environment variables for sensitive info in production
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'mlflow')
DB_USER = os.environ.get('DB_USER', 'mlflow')  # Don't hardcode in production
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'mlflow_password')  # Don't hardcode in production
MLFLOW_DB_URI = os.environ.get('MLFLOW_DB_URI', f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Training settings
USE_GPU = False  # Set to False for CPU optimization
```

Key changes:
- Environment variables for all configuration values with sensible defaults
- Cloud storage paths configured with S3 bucket references
- AWS credentials and region configuration
- Database connection parameters for PostgreSQL integration with RDS
- MLflow tracking server configuration

## 2. Amazon S3 Integration

### S3 Utilities (`src/utils/s3_utils.py`)

A new module was created to handle interactions with Amazon S3:

```python
def download_from_s3(s3_path, local_path):
    """
    Download a file from Amazon S3 to a local path.
    """
    bucket_name, key = parse_s3_path(s3_path)
    s3_client = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket_name, key, local_path)
    return local_path

def upload_to_s3(local_path, s3_path):
    """
    Upload a file from a local path to Amazon S3.
    """
    bucket_name, key = parse_s3_path(s3_path)
    s3_client = get_s3_client()
    s3_client.upload_file(local_path, bucket_name, key)
    return s3_path

def list_s3_files(s3_path, extension=None):
    """
    List all files in an S3 path, optionally filtered by extension.
    """
    bucket_name, prefix = parse_s3_path(s3_path)
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        return []
    
    files = []
    for obj in response['Contents']:
        if extension is None or obj['Key'].endswith(extension):
            files.append(f"s3://{bucket_name}/{obj['Key']}")
    
    return files
```

Key changes:
- Functions for downloading and uploading files to/from S3
- Helper functions to parse S3 paths (s3://bucket-name/path)
- File listing functionality with optional extension filtering
- Directory creation in S3 via marker files
- S3 client initialization with proper credentials handling

## 3. MLflow Integration

### MLflow Initialization Updates (`src/utils/mlflow_initialization.py`)

The MLflow initialization module was enhanced to support both GCP and AWS storage backends:

```python
# Try to import AWS configuration
try:
    from configs.aws_config import AWS_S3_BUCKET_NAME, MLFLOW_ARTIFACTS_PATH as AWS_MLFLOW_ARTIFACTS_PATH
    HAS_AWS_CONFIG = True
except ImportError:
    logger.warning("AWS configuration not found.")
    HAS_AWS_CONFIG = False

# Command line arguments for S3
parser.add_argument(
    "--use-s3",
    action="store_true",
    help="Use Amazon S3 for artifact storage."
)

# S3 artifact location configuration
if args.use_s3 and HAS_AWS_CONFIG:
    # Use the bucket from AWS config
    bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', AWS_S3_BUCKET_NAME)
    artifact_location = f"s3://{bucket_name}/{AWS_MLFLOW_ARTIFACTS_PATH}"
    logger.info(f"Using Amazon S3 for MLflow artifacts: {artifact_location}")
```

Key changes:
- Detection of AWS configuration availability
- Command-line arguments for S3 artifact storage
- Environment variable support for AWS S3 configuration
- S3 artifact location construction for MLflow

## 4. Prediction API Updates (`predict_api.py`)

The prediction API was updated to support loading models from both GCP and AWS cloud storage:

```python
# Check if Amazon S3 utilities are available
try:
    from src.utils.s3_utils import is_s3_path, download_from_s3, parse_s3_path
    HAS_S3_SUPPORT = True
except ImportError:
    logging.warning("Amazon S3 utilities not available. S3 features will be disabled.")
    HAS_S3_SUPPORT = False

# Cloud storage configuration
AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET_NAME', None)

# Handle S3 path for model loading
elif HAS_S3_SUPPORT and is_s3_path(checkpoint_path):
    logger.info(f"Downloading checkpoint from S3: {checkpoint_path}")
    # Create a temporary file to download the checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
        local_checkpoint_path = temp_file.name
    
    # Download the checkpoint from S3
    download_from_s3(checkpoint_path, local_checkpoint_path)
    logger.info(f"Downloaded checkpoint to {local_checkpoint_path}")

# Function to get model path (local, GCS, or S3)
def get_model_path(model_name):
    # First check local path
    local_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(local_path):
        return local_path
    
    # If not found locally and GCS is configured, check GCS
    if HAS_GCS_SUPPORT and GCS_BUCKET:
        gcs_path = f"gs://{GCS_BUCKET}/models/{model_name}"
        return gcs_path
    
    # If not found locally and S3 is configured, check S3
    if HAS_S3_SUPPORT and AWS_S3_BUCKET:
        s3_path = f"s3://{AWS_S3_BUCKET}/models/{model_name}"
        return s3_path
    
    # Default to local path if cloud storage not available
    return local_path
```

Key changes:
- Detection of S3 utilities availability
- S3 bucket configuration from environment variables
- Model path resolution that prioritizes local, then checks both GCS and S3
- Temporary file handling for S3 downloads
- Proper cleanup of temporary files after use

## 5. Dependencies Updates

Added AWS SDK for Python (boto3) to the requirements:

```
# Cloud Storage Integration
google-cloud-storage
boto3

# Database Integration
psycopg2-binary
```

## 6. Deployment Guide

A comprehensive AWS deployment guide (`docs/aws_deployment_guide.md`) was created with step-by-step instructions for:

1. Setting up S3 storage and uploading data
2. Deploying MLflow with RDS PostgreSQL and S3 integration
3. Setting up an EC2 instance for model training
4. Training models and tracking experiments with MLflow
5. Deploying the prediction API to AWS Fargate
6. Monitoring and maintenance
7. Resource cleanup

The guide includes detailed AWS CLI commands for each step, along with explanations of the AWS services being used.

## Summary of Changes

The project has been enhanced to:

1. Support both GCP and AWS as cloud providers
2. Use S3 for data and model storage
3. Leverage RDS PostgreSQL for MLflow backend
4. Deploy to EC2 and Fargate for compute resources
5. Utilize AWS monitoring and security features

These changes make the Flowers Classification App fully compatible with AWS services while maintaining the existing GCP functionality, allowing for flexible cloud deployment options.
