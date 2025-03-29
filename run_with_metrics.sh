#!/bin/bash
# Script to run training with metrics exposed to Prometheus

# Clean up any previous training container
docker rm -f training-metrics 2>/dev/null

echo "Starting training with metrics monitoring..."

# Run the container with a fixed name that Prometheus can discover
docker-compose run --name training-metrics app python src/train.py "$@"

echo "Training completed!"
