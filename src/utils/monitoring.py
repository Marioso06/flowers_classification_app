"""
Monitoring utilities for ML training and inference processes.
Provides Prometheus metrics exporters and helpers for model training.
"""
import os
import time
import threading
import logging
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Prometheus metrics collector for model training processes"""
    
    def __init__(self, port=8000):
        """
        Initialize the training monitor with Prometheus metrics
        
        Args:
            port (int): Port to expose Prometheus metrics on
        """
        self.port = port
        self.is_running = False
        
        # Define metrics
        self.epoch_counter = Counter('training_epochs_total', 'Total number of training epochs completed')
        self.batch_counter = Counter('training_batches_total', 'Total number of batches processed')
        
        self.train_loss = Gauge('training_loss', 'Current training loss')
        self.valid_loss = Gauge('validation_loss', 'Current validation loss')
        self.valid_accuracy = Gauge('validation_accuracy', 'Current validation accuracy')
        
        self.epoch_duration = Histogram('epoch_duration_seconds', 'Time taken to complete each epoch')
        self.batch_duration = Summary('batch_duration_seconds', 'Time taken to process each batch')
        
        # Resource metrics
        self.memory_usage = Gauge('training_memory_usage_bytes', 'Memory usage of the training process')
        self.cpu_usage = Gauge('training_cpu_usage_percent', 'CPU usage percentage of the training process')
        self.gpu_memory_usage = Gauge('training_gpu_memory_usage_bytes', 'GPU memory usage of the training process', ['device'])
        self.gpu_utilization = Gauge('training_gpu_utilization_percent', 'GPU utilization percentage', ['device'])
        
    def start(self):
        """Start the Prometheus HTTP server and resource monitoring thread"""
        try:
            # Start the Prometheus HTTP server
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start the resource monitoring thread
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring thread started")
            
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
            
    def stop(self):
        """Stop the resource monitoring thread"""
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Background thread to monitor system resources"""
        import torch
        
        while self.is_running:
            try:
                # Update CPU and memory metrics
                process = psutil.Process(os.getpid())
                self.memory_usage.set(process.memory_info().rss)
                self.cpu_usage.set(process.cpu_percent(interval=0.1))
                
                # Update GPU metrics if available
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # Get GPU memory usage
                        memory_allocated = torch.cuda.memory_allocated(i)
                        self.gpu_memory_usage.labels(device=f"cuda:{i}").set(memory_allocated)
                        
                        # Get GPU utilization - requires pynvml
                        try:
                            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
                            nvmlInit()
                            handle = nvmlDeviceGetHandleByIndex(i)
                            utilization = nvmlDeviceGetUtilizationRates(handle)
                            self.gpu_utilization.labels(device=f"cuda:{i}").set(utilization.gpu)
                        except ImportError:
                            logger.warning("pynvml not installed, GPU utilization not available")
                        except Exception as e:
                            logger.warning(f"Failed to get GPU utilization: {e}")
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring thread: {e}")
                time.sleep(30)  # Retry after a delay

    def record_epoch_start(self):
        """Record the start of a training epoch"""
        return time.time()
        
    def record_epoch_end(self, start_time, train_loss=None, valid_loss=None, valid_accuracy=None):
        """
        Record the end of a training epoch and update metrics
        
        Args:
            start_time: Time when the epoch started (from record_epoch_start)
            train_loss: Current training loss
            valid_loss: Current validation loss
            valid_accuracy: Current validation accuracy
        """
        self.epoch_counter.inc()
        duration = time.time() - start_time
        self.epoch_duration.observe(duration)
        
        if train_loss is not None:
            self.train_loss.set(train_loss)
        if valid_loss is not None:
            self.valid_loss.set(valid_loss)
        if valid_accuracy is not None:
            self.valid_accuracy.set(valid_accuracy)
        
    def record_batch(self, duration=None):
        """
        Record a processed batch
        
        Args:
            duration: Time taken to process the batch
        """
        self.batch_counter.inc()
        if duration is not None:
            self.batch_duration.observe(duration)

# Singleton instance for easy import
def get_training_monitor(port=8000):
    """Get or create the training monitor singleton instance"""
    if not hasattr(get_training_monitor, 'instance'):
        get_training_monitor.instance = TrainingMonitor(port=port)
    return get_training_monitor.instance
