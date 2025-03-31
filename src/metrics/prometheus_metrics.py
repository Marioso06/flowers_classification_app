"""
Prometheus metrics and monitoring utilities for flower classification app
"""
import time
import logging
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logger = logging.getLogger(__name__)

# Custom metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

# Explanation metrics
explanation_requests = Counter('model_explanation_requests_total', 'Total number of explanation requests', ['explanation_type', 'model_version'])
explanation_time = Histogram('model_explanation_duration_seconds', 'Time spent generating explanations', ['explanation_type', 'model_version'])
explanation_failures = Counter('model_explanation_failures_total', 'Total number of explanation failures', ['explanation_type', 'model_version'])

def monitor_resources(interval=5):
    """
    Monitor system resources and update Prometheus metrics
    
    Args:
        interval: How often to update metrics in seconds
    """
    logger.info(f"Starting resource monitoring with interval {interval}s")
    
    while True:
        try:
            # Update memory usage (bytes)
            process = psutil.Process()
            mem_info = process.memory_info()
            memory_usage.set(mem_info.rss)
            
            # Update CPU usage (percent)
            cpu_percent = process.cpu_percent(interval=0.1)
            cpu_usage.set(cpu_percent)
            
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            time.sleep(interval)  # Still sleep to avoid tight loops on error

def start_monitoring_thread():
    """Start the resource monitoring in a background thread"""
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    logger.info("Resource monitoring thread started")
    return monitor_thread
