# GCP Code Changes Explanation

This document provides a comprehensive explanation of the code changes made to adapt the Flowers Classification App for deployment on Google Cloud Platform (GCP). The original project was designed to run locally, and several modifications were needed to make it cloud-ready and take advantage of GCP services.

## 1. Configuration Changes

### GCP Configuration File (`configs/gcp_config.py`)

A dedicated configuration file was created to centralize all GCP-related settings:

```python
import os

# GCP Configuration
GCP_BUCKET_NAME = os.environ.get('BUCKET_NAME', 'flowers-classification-lab-0147')
GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')

# Storage paths
DATA_PATH = f"gs://{GCP_BUCKET_NAME}/data"
MODEL_PATH = f"gs://{GCP_BUCKET_NAME}/models"
OUTPUT_PATH = f"gs://{GCP_BUCKET_NAME}/outputs"

# MLflow settings
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_ARTIFACTS_PATH = os.environ.get('MLFLOW_ARTIFACTS_PATH', 'mlflow-artifacts')
MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'flowers-classification')

# Database settings - use environment variables for sensitive info in production
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'mlflow')
DB_USER = os.environ.get('DB_USER', 'mlflow')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'mlflow_password')
MLFLOW_DB_URI = os.environ.get('MLFLOW_DB_URI', f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Training settings
USE_GPU = False  # Set to False for CPU optimization
```

Key changes:
- Environment variables for all configuration values with sensible defaults
- Cloud storage paths configured with GCS bucket references
- Database connection parameters for PostgreSQL integration
- MLflow tracking server configuration

## 2. Google Cloud Storage Integration

### GCS Utilities (`src/utils/gcs_utils.py`)

A new module was created to handle interactions with Google Cloud Storage:

```python
def download_from_gcs(gcs_path, local_path):
    """
    Download a file from Google Cloud Storage to a local path.
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(local_path, gcs_path):
    """
    Upload a file from a local path to Google Cloud Storage.
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return gcs_path

def list_gcs_files(gcs_path, extension=None):
    """
    List all files in a GCS path, optionally filtered by extension.
    """
    bucket_name, prefix = parse_gcs_path(gcs_path)
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    files = []
    for blob in blobs:
        if extension is None or blob.name.endswith(extension):
            files.append(f"gs://{bucket_name}/{blob.name}")
    
    return files
```

Key changes:
- Functions for downloading and uploading files to/from GCS
- Helper functions to parse GCS paths (gs://bucket-name/path)
- File listing functionality with optional extension filtering
- Directory creation in GCS via marker files

## 3. MLflow Integration

### MLflow Initialization (`src/utils/mlflow_initialization.py`)

The MLflow integration was enhanced to support cloud-based backends and artifact storage:

```python
def main():
    args = parse_arguments()
    
    # Check for environment variables that might override arguments
    if 'MLFLOW_HOST' in os.environ:
        args.host = os.environ['MLFLOW_HOST']
    
    if 'MLFLOW_PORT' in os.environ:
        args.port = int(os.environ['MLFLOW_PORT'])
    
    # If PostgreSQL is requested, set the backend store URI from config/env
    if args.use_postgres:
        if not HAS_POSTGRES:
            logger.warning("PostgreSQL support requested but psycopg2 not installed.")
            
        # Get DB URI from environment variable or config
        if 'MLFLOW_DB_URI' in os.environ:
            args.backend_store_uri = os.environ['MLFLOW_DB_URI']
        elif HAS_GCP_CONFIG and hasattr(MLFLOW_DB_URI):
            args.backend_store_uri = MLFLOW_DB_URI
    
    # Check if GCS should be used for artifact storage
    if args.use_gcs and HAS_GCP_CONFIG:
        # Use the bucket from GCP config
        bucket_name = os.environ.get('BUCKET_NAME', GCP_BUCKET_NAME)
        artifact_location = f"gs://{bucket_name}/{MLFLOW_ARTIFACTS_PATH}"
        logger.info(f"Using Google Cloud Storage for MLflow artifacts: {artifact_location}")
```

Key changes:
- Command line arguments for configuring PostgreSQL as backend store
- GCS artifact storage integration
- Environment variable overrides for configuration
- Automatic detection of required dependencies
- Production-ready configuration warnings (e.g., not using SQLite in production)

## 4. Prediction API Enhancements

### API Modifications (`predict_api.py`)

The prediction API was enhanced to support cloud-based model storage and deployment:

```python
# Check if Google Cloud Storage utilities are available
try:
    from src.utils.gcs_utils import is_gcs_path, download_from_gcs, parse_gcs_path
    HAS_GCS_SUPPORT = True
except ImportError:
    logging.warning("Google Cloud Storage utilities not available. GCS features will be disabled.")
    HAS_GCS_SUPPORT = False

# Check if MLflow is available for model loading
try:
    import mlflow.pyfunc
    HAS_MLFLOW_SUPPORT = True
except ImportError:
    logging.warning("MLflow not available. MLflow features will be disabled.")
    HAS_MLFLOW_SUPPORT = False

# GCS configuration
GCS_BUCKET = os.environ.get('GCS_BUCKET', None)  # Set this in your environment or docker-compose
if GCS_BUCKET and HAS_GCS_SUPPORT:
    logger.info(f"Google Cloud Storage bucket configured: {GCS_BUCKET}")
else:
    logger.info("Using local storage only for model files")

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
USE_MLFLOW_MODELS = os.environ.get('USE_MLFLOW_MODELS', 'false').lower() in ('true', '1', 'yes')
```

Key changes:
- Conditional imports for cloud features (GCS, MLflow)
- Environment variable configuration for different deployment scenarios
- Dual-path model loading (direct files or via MLflow)
- Improved error handling and logging for cloud environments

## 5. Dependency Management

### Updated Requirements (`requirements.txt`)

Additional dependencies were added to the project to support cloud deployment:

```
matplotlib==3.10.0
Pillow==11.1.0
PyYAML==6.0.2
gdown==5.2.0
dvc==3.59.0
dvc-gdrive==3.0.1
mlflow==2.20.2
Flask==3.1.0

# Cloud Storage Integration
google-cloud-storage>=2.10.0

# Database Integration
psycopg2-binary>=2.9.9  # PostgreSQL connector

# For production deployment
gunicorn>=21.2.0
```

Key additions:
- Google Cloud Storage client library
- PostgreSQL connector for database backend
- Gunicorn for production-grade HTTP serving
- Pinned versions for stability

## 6. Training Script Modifications

The training script was modified to support cloud-based data and model storage:

- Model saving to GCS instead of local storage
- Loading datasets from GCS when available
- Integration with MLflow for experiment tracking
- Support for both local and cloud training environments

## 7. Container and Deployment Configuration

### Dockerfile and Cloud Run Configuration

A production-ready Dockerfile was created with specific optimizations for Cloud Run:

- Python 3.10 slim base image for reduced size
- CPU-optimized PyTorch installation
- Environment variable configuration
- Gunicorn server for production deployment
- Integration with GCS and MLflow
- Proper handling of temporary files

## 8. Security Enhancements

- Environment variables for all sensitive information
- Secret Manager integration for database credentials
- Optional authentication for Cloud Run services
- Secure handling of service accounts and permissions

## Summary of Architectural Changes

1. **Storage Layer**: Migrated from local file system to Google Cloud Storage
   - Added transparent path handling (local vs. gs:// paths)
   - Implemented utility functions for GCS operations

2. **Database Layer**: Enhanced MLflow to use PostgreSQL instead of SQLite
   - Added Cloud SQL integration for production-ready database
   - Implemented proper connection handling and error recovery

3. **Compute Layer**: 
   - Added VM deployment for training
   - Configured Cloud Run for prediction API

4. **Experiment Tracking**:
   - Enhanced MLflow to use cloud-based backends and artifact storage
   - Added proper experiment naming and versioning

5. **Deployment Pipeline**:
   - Created Docker containers optimized for cloud deployment
   - Added environment-specific configuration

These changes have transformed the application from a local-only solution to a scalable, cloud-native application that leverages GCP services for improved performance, reliability, and maintainability.
