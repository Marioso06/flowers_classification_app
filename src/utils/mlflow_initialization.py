import argparse
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import GCP configuration
try:
    from configs.gcp_config import GCP_BUCKET_NAME, MLFLOW_ARTIFACTS_PATH, MLFLOW_DB_URI
    HAS_GCP_CONFIG = True
except ImportError:
    logger.warning("GCP configuration not found. Using local storage for MLflow.")
    HAS_GCP_CONFIG = False
    
# Try to import database libraries (required for PostgreSQL)
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    logger.warning("PostgreSQL dependencies not found. If using PostgreSQL backend, install psycopg2.")
    HAS_POSTGRES = False

def parse_arguments():
    """Parse command line arguments for configuring the MLflow server."""
    parser = argparse.ArgumentParser(description="Initialize an MLflow Tracking Server.")
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The IP address to bind to (default: 127.0.0.1)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="The port to bind to (default: 8080)."
    )
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="The URI of the backend store (e.g., sqlite:///mlflow.db or postgresql://user:password@localhost/mlflow)."
    )
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Use PostgreSQL as the backend store (requires MLFLOW_DB_URI in environment or config)."
    )
    parser.add_argument(
        "--artifact-store",
        type=str,
        help="The artifact store location (local path or GCS bucket path)."
    )
    parser.add_argument(
        "--use-gcs",
        action="store_true",
        help="Use Google Cloud Storage for artifact storage."
    )
    return parser.parse_args()

def start_mlflow_server(args):
    """Start the MLflow Tracking Server with the specified arguments."""
    command = [
        "mlflow", "server",
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    # Add backend store URI if provided
    if args.backend_store_uri:
        command.extend(["--backend-store-uri", args.backend_store_uri])
    
    # Set up artifact storage
    artifact_location = None
    
    # Check if GCS should be used
    if args.use_gcs and HAS_GCP_CONFIG:
        # Use the bucket from GCP config
        bucket_name = os.environ.get('BUCKET_NAME', GCP_BUCKET_NAME)
        artifact_location = f"gs://{bucket_name}/{MLFLOW_ARTIFACTS_PATH}"
        logger.info(f"Using Google Cloud Storage for MLflow artifacts: {artifact_location}")
    elif args.artifact_store:
        # Use explicitly provided artifact store
        artifact_location = args.artifact_store
        logger.info(f"Using custom artifact location: {artifact_location}")
    
    # Add artifact location to command if set
    if artifact_location:
        command.extend(["--default-artifact-root", artifact_location])
    
    logger.info(f"Starting MLflow server with command: {' '.join(command)}")
    
    try:
        # Run in background for use with Docker
        if os.environ.get('MLFLOW_DOCKER_MODE') == 'true':
            subprocess.Popen(command)
            logger.info("MLflow server started in background mode")
        else:
            # Run in foreground (default)
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting MLflow server: {e}")
        exit(1)

def main():
    args = parse_arguments()
    
    # Check for environment variables that might override arguments
    if 'MLFLOW_HOST' in os.environ:
        args.host = os.environ['MLFLOW_HOST']
    
    if 'MLFLOW_PORT' in os.environ:
        args.port = int(os.environ['MLFLOW_PORT'])
    
    if 'MLFLOW_BACKEND_STORE_URI' in os.environ:
        args.backend_store_uri = os.environ['MLFLOW_BACKEND_STORE_URI']
    
    if 'MLFLOW_ARTIFACT_STORE' in os.environ:
        args.artifact_store = os.environ['MLFLOW_ARTIFACT_STORE']
    
    if 'USE_GCS_FOR_MLFLOW' in os.environ and os.environ['USE_GCS_FOR_MLFLOW'].lower() in ('true', '1', 'yes'):
        args.use_gcs = True
        
    if 'USE_POSTGRES_FOR_MLFLOW' in os.environ and os.environ['USE_POSTGRES_FOR_MLFLOW'].lower() in ('true', '1', 'yes'):
        args.use_postgres = True
        
    # If PostgreSQL is requested, set the backend store URI from config/env
    if args.use_postgres:
        if not HAS_POSTGRES:
            logger.warning("PostgreSQL support requested but psycopg2 not installed. Try: pip install psycopg2-binary")
            
        # Get DB URI from environment variable or config
        if 'MLFLOW_DB_URI' in os.environ:
            args.backend_store_uri = os.environ['MLFLOW_DB_URI']
        elif HAS_GCP_CONFIG and hasattr(MLFLOW_DB_URI):
            args.backend_store_uri = MLFLOW_DB_URI
        else:
            logger.warning("PostgreSQL backend requested but no database URI found. Check environment variables or config.")
    
    # Create directories if needed for SQLite
    if args.backend_store_uri and args.backend_store_uri.startswith('sqlite:///') and not args.backend_store_uri.startswith('sqlite:///:memory:'):
        db_path = args.backend_store_uri.replace('sqlite:///', '')
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        logger.info(f"Ensuring SQLite database directory exists: {os.path.dirname(os.path.abspath(db_path))}")
    
    # Log database backend type
    if args.backend_store_uri.startswith('postgresql://'):
        logger.info("Using PostgreSQL as backend store (production ready)")
    elif args.backend_store_uri.startswith('mysql://'):
        logger.info("Using MySQL as backend store (production ready)")
    elif args.backend_store_uri.startswith('sqlite:///'):
        logger.warning("Using SQLite as backend store (not recommended for production)")
    else:
        logger.info(f"Using custom backend store: {args.backend_store_uri.split('://')[0]}")
    
    # Start the MLflow server
    start_mlflow_server(args)

if __name__ == "__main__":
    main()