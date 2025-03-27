"""
Test script to generate and expose training metrics for Prometheus.
This script simulates a training process to help debug metric collection.
"""
import time
import logging
import random
from src.utils.monitoring import get_training_monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_training():
    """Simulate a model training process with metrics"""
    # Initialize the training monitor
    monitor = get_training_monitor(port=8000)
    monitor.start()
    logger.info("Training metrics server started on port 8000")
    logger.info("Access metrics at: http://localhost:8000/metrics")
    
    try:
        # Simulate training for 30 minutes (or until interrupted)
        epoch = 0
        total_batches = 0
        logger.info("Starting simulated training process...")
        
        while epoch < 10:  # Simulate 10 epochs
            epoch_start = monitor.record_epoch_start()
            logger.info(f"Starting epoch {epoch+1}/10")
            
            # Simulate batch processing
            for batch in range(20):  # 20 batches per epoch
                batch_start = time.time()
                
                # Simulate batch processing time
                time.sleep(0.2)  # 200ms per batch
                
                # Update batch metrics
                total_batches += 1
                monitor.record_batch(duration=time.time() - batch_start)
                
                # Simulate batch loss
                batch_loss = 1.0 - 0.05 * epoch - 0.005 * batch + random.uniform(-0.02, 0.02)
                logger.info(f"  Batch {batch+1}/20 completed, loss: {batch_loss:.4f}")
            
            # Simulate validation at end of epoch
            time.sleep(0.5)  # Validation time
            
            # Calculate simulated metrics
            train_loss = 1.0 - 0.06 * epoch + random.uniform(-0.01, 0.01)
            valid_loss = 1.0 - 0.05 * epoch + random.uniform(-0.02, 0.02)
            valid_accuracy = 0.70 + 0.025 * epoch + random.uniform(-0.01, 0.01)
            
            # Update epoch metrics
            monitor.record_epoch_end(
                epoch_start,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_accuracy=valid_accuracy
            )
            
            logger.info(f"Epoch {epoch+1}/10 completed:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Valid Loss: {valid_loss:.4f}")
            logger.info(f"  Valid Accuracy: {valid_accuracy:.4f}")
            
            epoch += 1
            
        logger.info("Simulated training completed!")
        # Keep the server running so metrics remain available
        logger.info("Keeping metrics server running. Press Ctrl+C to exit.")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Training simulation stopped by user")
    finally:
        monitor.stop()
        logger.info("Metrics server stopped")

if __name__ == "__main__":
    simulate_training()
