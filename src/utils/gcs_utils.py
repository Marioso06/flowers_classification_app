"""
Google Cloud Storage Utilities

This module provides helper functions for interacting with Google Cloud Storage
from the Flowers Classification App.
"""
from google.cloud import storage
import os
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_gcs_path(path):
    """Check if a path is a GCS path (starts with gs://)."""
    return path.startswith("gs://")

def parse_gcs_path(gcs_path):
    """Parse a GCS path into bucket name and blob name."""
    if not is_gcs_path(gcs_path):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
    
    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    return bucket_name, blob_name

def download_from_gcs(gcs_path, local_path):
    """
    Download a file from Google Cloud Storage to a local path.
    
    Args:
        gcs_path: The GCS path (gs://bucket-name/path/to/file)
        local_path: The local destination path
        
    Returns:
        The local path where the file was downloaded
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Create the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Make sure the local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download
    logger.info(f"Downloading {gcs_path} to {local_path}")
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(local_path, gcs_path):
    """
    Upload a file from a local path to Google Cloud Storage.
    
    Args:
        local_path: The local source path
        gcs_path: The GCS destination path (gs://bucket-name/path/to/file)
        
    Returns:
        The GCS path where the file was uploaded
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Create the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload
    logger.info(f"Uploading {local_path} to {gcs_path}")
    blob.upload_from_filename(local_path)
    return gcs_path

def list_gcs_files(gcs_path, extension=None):
    """
    List all files in a GCS path, optionally filtered by extension.
    
    Args:
        gcs_path: The GCS path to list (gs://bucket-name/path)
        extension: Optional file extension filter (e.g., '.jpg')
        
    Returns:
        List of full GCS paths to the matching files
    """
    bucket_name, prefix = parse_gcs_path(gcs_path)
    
    # Create the client
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    # Filter by extension if provided
    if extension:
        return [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(extension)]
    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]

def create_gcs_directory(gcs_path):
    """
    Create a directory-like structure in GCS by uploading an empty marker file.
    
    Args:
        gcs_path: The GCS path to create (gs://bucket-name/path/)
        
    Returns:
        The GCS path of the marker file
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Ensure the path ends with a slash and add a marker file
    if not blob_name.endswith('/'):
        blob_name += '/'
    blob_name += '.keep'
    
    # Create the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Create an empty file
    with tempfile.NamedTemporaryFile() as temp:
        blob.upload_from_filename(temp.name)
    
    return f"gs://{bucket_name}/{blob_name}"
