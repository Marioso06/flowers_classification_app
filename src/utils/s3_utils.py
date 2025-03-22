"""
Amazon S3 Utilities

This module provides helper functions for interacting with Amazon S3
from the Flowers Classification App.
"""
import boto3
import os
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_s3_path(path):
    """Check if a path is an S3 path (starts with s3://)."""
    return path.startswith("s3://")

def parse_s3_path(s3_path):
    """Parse an S3 path into bucket name and key."""
    if not is_s3_path(s3_path):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")
    
    path_parts = s3_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""
    
    return bucket_name, key

def get_s3_client():
    """Get an S3 client using credentials from environment variables."""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

def download_from_s3(s3_path, local_path):
    """
    Download a file from Amazon S3 to a local path.
    
    Args:
        s3_path: The S3 path (s3://bucket-name/path/to/file)
        local_path: The local destination path
        
    Returns:
        The local path where the file was downloaded
    """
    bucket_name, key = parse_s3_path(s3_path)
    
    # Create the client
    s3_client = get_s3_client()
    
    # Make sure the local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download
    logger.info(f"Downloading {s3_path} to {local_path}")
    s3_client.download_file(bucket_name, key, local_path)
    return local_path

def upload_to_s3(local_path, s3_path):
    """
    Upload a file from a local path to Amazon S3.
    
    Args:
        local_path: The local source path
        s3_path: The S3 destination path (s3://bucket-name/path/to/file)
        
    Returns:
        The S3 path where the file was uploaded
    """
    bucket_name, key = parse_s3_path(s3_path)
    
    # Create the client
    s3_client = get_s3_client()
    
    # Upload
    logger.info(f"Uploading {local_path} to {s3_path}")
    s3_client.upload_file(local_path, bucket_name, key)
    return s3_path

def list_s3_files(s3_path, extension=None):
    """
    List all files in an S3 path, optionally filtered by extension.
    
    Args:
        s3_path: The S3 path to list (s3://bucket-name/path)
        extension: Optional file extension filter (e.g., '.jpg')
        
    Returns:
        List of full S3 paths to the matching files
    """
    bucket_name, prefix = parse_s3_path(s3_path)
    
    # Create the client
    s3_client = get_s3_client()
    
    # List objects
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Check if there are contents
    if 'Contents' not in response:
        return []
    
    # Filter by extension if provided
    if extension:
        return [f"s3://{bucket_name}/{obj['Key']}" for obj in response['Contents'] if obj['Key'].endswith(extension)]
    return [f"s3://{bucket_name}/{obj['Key']}" for obj in response['Contents']]

def create_s3_directory(s3_path):
    """
    Create a directory-like structure in S3 by uploading an empty marker file.
    
    Args:
        s3_path: The S3 path to create (s3://bucket-name/path/)
        
    Returns:
        The S3 path of the marker file
    """
    bucket_name, key = parse_s3_path(s3_path)
    
    # Ensure the path ends with a slash and add a marker file
    if not key.endswith('/'):
        key += '/'
    key += '.keep'
    
    # Create the client
    s3_client = get_s3_client()
    
    # Create an empty file
    with tempfile.NamedTemporaryFile() as temp:
        s3_client.upload_file(temp.name, bucket_name, key)
    
    return f"s3://{bucket_name}/{key}"
