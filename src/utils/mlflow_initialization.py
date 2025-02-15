import argparse
import subprocess
import os

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
        help="The URI of the backend store (e.g., sqlite:///mlflow.db or postgresql://user:password@localhost/mlflow)."
    )
    return parser.parse_args()

def start_mlflow_server(args):
    """Start the MLflow Tracking Server with the specified arguments."""
    command = [
        "mlflow", "server",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    print(f"Starting MLflow server with command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        exit(1)

def main():
    args = parse_arguments()
    start_mlflow_server(args)

if __name__ == "__main__":
    main()