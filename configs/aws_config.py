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
