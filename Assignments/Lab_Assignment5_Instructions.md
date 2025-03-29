# Lab Assignment 5: Monitoring and Observability with Prometheus and Grafana

## Overview

In this lab assignment, you will enhance your containerized machine learning application by integrating Prometheus and Grafana for monitoring and observability. This is a critical aspect of MLOps that allows you to:

* **Monitor model training progress** in real-time
* **Track prediction API performance** and usage patterns
* **Identify system bottlenecks** and resource constraints
* **Set up alerts** for potential issues
* **Visualize key metrics** through custom dashboards

By the end of this lab, you will have a comprehensive monitoring solution that provides insights into both your machine learning processes and the underlying infrastructure.

## Monitoring and Observability Fundamentals

### What is Prometheus?

Prometheus is an open-source systems monitoring and alerting toolkit that collects and stores metrics as time-series data. Key features include:

* **Pull-based architecture**: Prometheus scrapes metrics from targets at specified intervals
* **Flexible data model**: Uses a multi-dimensional data model with key-value pairs
* **Built-in query language**: PromQL for sophisticated querying and aggregation
* **No reliance on distributed storage**: Stores data locally and is autonomous
* **Target discovery**: Supports service discovery to identify targets dynamically

### What is Grafana?

Grafana is an open-source visualization and analytics software that allows you to:

* Create dynamic, reusable dashboards
* Visualize metrics from various data sources (including Prometheus)
* Set up alerting rules based on metrics
* Annotate and correlate events
* Share dashboards across teams

### The Importance of Monitoring ML Systems

Machine learning systems have unique monitoring requirements beyond traditional applications:

1. **Model Performance Tracking**: Monitor accuracy, loss, and other ML-specific metrics during training and inference
2. **Data Drift Detection**: Identify when input data patterns change, potentially affecting model performance
3. **Resource Utilization**: Track GPU, CPU, and memory usage during computationally intensive training jobs
4. **Inference Latency**: Measure response times for real-time prediction requests
5. **Pipeline Health**: Monitor the entire ML workflow from data ingestion to prediction

Implementing proper monitoring allows you to:
* Detect and debug issues early
* Optimize resource allocation
* Understand training dynamics
* Ensure reliable prediction services
* Make data-driven decisions about model retraining

## Assignment Instructions

### 1. Set Up Monitoring for Your Model Training Process

First, you'll add monitoring capabilities to your training script using Prometheus metrics.

#### Create a Monitoring Utility Module for Training Processes

Monitoring training processes differs significantly from monitoring API services, requiring a separate approach. While the API serves continuous requests with relatively stable resource usage, training processes are typically:

- **Ephemeral**: Training jobs run for a limited time and then complete
- **Resource-intensive**: They often utilize maximum available CPU/GPU resources
- **Batch-oriented**: They process data in epochs and batches rather than individual requests
- **Progressive**: They report metrics that should improve over time (loss decreasing, accuracy increasing)

For these reasons, we'll create a dedicated monitoring module for training processes. This module will:

1. Start its own HTTP server to expose Prometheus metrics (since training often runs separately from the API)
2. Track training-specific metrics like epochs, batches, loss values, and validation metrics
3. Include resource monitoring appropriate for ML training (including GPU metrics if available)
4. Provide helper methods tailored to the training workflow

##### Key Differences Between Training and API Monitoring

| Aspect | Training Monitoring | API Monitoring |
|--------|--------------------|-----------------|
| **Lifetime** | Temporary (runs during training job) | Persistent (runs as long as service is up) |
| **Metrics Focus** | Model performance metrics (loss, accuracy) | Service metrics (requests, latency) |
| **Resource Usage** | Typically high and sustained | Variable based on request load |
| **Implementation** | Standalone HTTP server | Integrated with Flask app |
| **Data Pattern** | Progressive metrics showing improvement | Stability metrics showing consistency |
| **GPU Monitoring** | Critical - often the performance bottleneck | Optional - may not use GPU for inference |
| **Usage Pattern** | Used to track active experiments | Used to monitor production service health |

While both use Prometheus as the underlying metrics system, they serve different purposes in the ML lifecycle: training monitoring helps optimize model development, while API monitoring ensures reliable serving in production.

#### Monitoring for Different Model Types

As many students are using regression and decision tree-based models rather than neural networks. For these models, the monitoring approach differs slightly:

**For Regression Models:**
- Focus on metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² values
- Monitor feature importance to understand which inputs drive predictions
- Track regularization parameters (like alpha in Ridge/Lasso) during hyperparameter tuning

**For Decision Tree-Based Models (Random Forest, XGBoost, etc.):**
- Monitor depth and complexity metrics (tree depth, number of nodes)
- Track feature importance and how it changes during training
- For boosting models, monitor per-iteration improvements
- Track overfitting indicators (training vs. validation performance gap)

Create a file named `monitoring.py` in your project's utils directory:

**Example for Regression Monitoring:**

```python
# Add these metrics to your monitoring.py for regression models
class RegressionMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)
        
        # Regression-specific metrics
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')
        
        # Feature importance tracking (top 5 features)
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])
        
    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        """Record regression metrics"""
        if mse is not None:
            self.mse.set(mse)
        if rmse is not None:
            self.rmse.set(rmse)
        if mae is not None:
            self.mae.set(mae)
        if r_squared is not None:
            self.r_squared.set(r_squared)
            
        # Update feature importance for top features
        if feature_importance is not None:
            # Sort features by importance (assuming dict format)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)
```

**Example for Tree-Based Models:**

```python
# Add these metrics to your monitoring.py for decision tree models
class TreeModelMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)
        
        # Tree metrics
        self.tree_depth = Gauge('tree_max_depth', 'Maximum tree depth')
        self.tree_leaves = Gauge('tree_leaf_count', 'Number of leaf nodes')
        self.trees_count = Gauge('ensemble_tree_count', 'Number of trees in the ensemble')
        
        # Boosting iteration counters
        self.boost_round = Counter('boosting_rounds_total', 'Total boosting rounds completed')
        self.iteration_improvement = Gauge('iteration_improvement', 'Performance improvement in the last iteration')
                
    def record_tree_metrics(self, depth=None, leaves=None, trees=None):
        """Record tree structure metrics"""
        if depth is not None:
            self.tree_depth.set(depth)
        if leaves is not None:
            self.tree_leaves.set(leaves)
        if trees is not None:
            self.trees_count.set(trees)
            
    def record_boost_round(self, improvement=None):
        """Record a completed boosting round"""
        self.boost_round.inc()
        if improvement is not None:
            self.iteration_improvement.set(improvement)
```

Adapt these examples to your specific models and the metrics most relevant to your ML task.


### 2. Add Monitoring to Your Prediction API

For your Flask prediction API, you'll want to monitor different metrics related to inference performance. The best approach is to use the `prometheus_flask_exporter` package, which integrates directly with Flask. You can follow this template for your convenience:

```python
# In your predict_api.py or similar file
from flask import Flask, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time

app = Flask(__name__)

# Initialize Prometheus metrics - this automatically exposes metrics at /metrics
metrics = PrometheusMetrics(app)

# Custom metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

# In your prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    model_version = "1.0"  # Replace with your actual version tracking
    
    try:
        # Your existing prediction code
        # ...
        
        # Record successful prediction
        prediction_requests.labels(
            model_version=model_version,
            status="success"
        ).inc()
        
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)
        
        return jsonify(result)
        
    except Exception as e:
        # Record failed prediction
        prediction_requests.labels(
            model_version=model_version,
            status="error"
        ).inc()
        
        return jsonify({"error": str(e)}), 500

# Add a function to monitor resource usage in the background
def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    import psutil
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

# Start resource monitoring in a background thread when app starts
if __name__ == "__main__":
    import threading
    import os
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
```

With this approach, your Flask application will automatically expose metrics at the `/metrics` endpoint (typically at port 5000 or whatever port your Flask app is running on). This endpoint will be scraped by Prometheus as configured in your `prometheus.yml` file.

#### Understanding the Background Monitoring Thread

The `monitor_thread` is a crucial component of your monitoring solution, and it's worth explaining how it works and why it's necessary:

1. **Purpose**: The monitoring thread continuously collects system resource metrics (memory, CPU) in the background while your API is running. This provides real-time visibility into your application's resource consumption without impacting the request-response cycle.

2. **Implementation**: 
   - It's created as a daemon thread, meaning it will automatically terminate when the main application exits
   - It runs in a separate execution path from your Flask routes
   - It updates Prometheus gauge metrics at a regular interval (every 15 seconds - This can be changed)
   - It uses `psutil` to collect system resource information in a cross-platform way

3. **Benefits**:
   - **Non-blocking**: Resource monitoring happens independently of API requests
   - **Continuous measurement**: Provides data even when there are no active requests
   - **Early warning system**: Can detect resource issues before they impact API performance
   - **Low overhead**: The 15-second interval provides good visibility with minimal impact

4. **Alternative approaches**:
   - You could monitor resources on each API call, but this would miss data between calls
   - You could use a separate process, but that adds complexity to your deployment
   - You could rely on container-level metrics, but they might not provide application-specific details

Implementing resource monitoring with a background thread is a pattern commonly used in production ML systems, where you need to track both model performance (accuracy, latency) and system performance (memory, CPU) to ensure reliable operation as we saw in Unit 9.

### 3. Set Up Prometheus Configuration

Create a directory for Prometheus configuration:

```bash
mkdir -p prometheus
```

or create the folder within your IDE.

Create a `prometheus.yml` file in this directory:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    scrape_interval: 5s
    static_configs:
      - targets: ['app:9000']

  - job_name: 'prometheus'
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:9090']
      
  - job_name: 'model-training' (Watch the video that explains how to set up model training metrics)
    scrape_interval: 5s
    static_configs:
      - targets:
        - 'training-metrics:8002'
    metrics_path: /metrics
    scheme: http
    scrape_timeout: 3s
```

### 4. Update Docker Compose Configuration

#### Using Official Images for Monitoring Tools

Unlike your ML application and MLflow, we don't need to create custom Dockerfiles for Prometheus and Grafana. Instead, we'll use the official images directly in the docker-compose.yml file. Here's why:

1. **Standardization**: Prometheus and Grafana provide official, well-maintained Docker images that follow best practices
2. **Simplicity**: The official images require minimal configuration to get started
3. **Reliability**: These images are regularly updated with security patches and performance improvements
4. **Customization**: We can still customize behavior through volume mounts and environment variables

This approach of using official images for standard components while creating custom Dockerfiles only for your application code is a common pattern in containerized deployments. It reduces maintenance overhead and leverages the expertise of the tool maintainers.

Modify your `docker-compose.yml` to include Prometheus and Grafana service, follow the template below:

```yaml
version: '3'

services:
  app:
    ... <same as you have in your previous submission>

  mlflow:
    ... <same as you have in your previous submission>

  prometheus:
    <Include the prometheus service configuration found in the Flowers Classification App repo - You can make changes if necessary>

  grafana:
    <Include the grafana service configuration found in the Flowers Classification App repo - You can make changes if necessary>

networks:
  ml-network:
    driver: bridge

volumes:
  mlflow-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
```

### 5. Create Basic Grafana Dashboards

#### Understanding Grafana Provisioning

Grafana supports two ways to configure data sources and dashboards:

1. **Manual configuration through the Web UI** (interactive but not reproducible)
2. **Automated provisioning through files** (reproducible and version-controlled)

For production deployments and containerized applications, the file-based provisioning approach offers several advantages:

* **Reproducibility**: Configuration is consistent across environments
* **Version control**: Configuration files can be tracked in Git, just like your code
* **Automation**: No manual setup steps when deploying in new environments
* **Infrastructure as Code**: Follows DevOps best practices

While the Web UI is more intuitive for exploration and development, provisioning files align with the containerization principles we've been learning throughout these labs.

#### Set up the Prometheus data source:

Create `grafana/provisioning/datasources/prometheus.yml`. You can follow the file provided in the Flowers Classification App repo. For example:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Alternative Web UI approach**: You could alternatively configure this after Grafana is running:
1. Navigate to http://localhost:3000
2. Log in with admin/password
3. Go to Configuration → Data Sources → Add data source
4. Select Prometheus
5. Set the URL to http://prometheus:9090
6. Click Save & Test

However, this manual configuration would be lost each time your container is rebuilt, making the file-based approach more suitable for containerized deployments. For instance, this is not advisable for your submission and it is only for your knowledge and training.

#### Create a basic training dashboard:

Create `grafana/provisioning/dashboards/training_dashboard.json`. You can see the explorer created in the `flowers_classification_app` repo or just follow the examples provided in the [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/).

#### Create a basic predict API dashboard:

Create `grafana/provisioning/dashboards/predict_api_dashboard.json`. You can see the explorer created in the `flowers_classification_app` repo or just follow the examples provided in the [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/).

#### Create your dashboard configuration file:

Create `grafana/provisioning/dashboards/dashboard.yml`. This file is essential because it tells Grafana:

1. **Where to find** your dashboard JSON files
2. **How to organize** them (folder structure)
3. **Update behavior** (what happens when dashboards change)
4. **User permissions** for the provisioned dashboards

Example dashboard configuration:

```yaml
apiVersion: 1

providers:
  - name: 'Default'
    orgId: 1
    folder: 'ML Monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
```

Without this configuration file, Grafana wouldn't know how to load your dashboard JSON files during startup. This is a key component of the provisioning system that enables fully automated, zero-touch dashboard setup.

### 6. Configure Prometheus Alerts

Monitoring is most valuable when it can proactively notify you of issues before they become critical failures. Prometheus provides a powerful alerting system that can trigger notifications based on specific metric conditions.

#### Why Alerting is Critical for ML Systems

ML applications have unique alert requirements beyond traditional applications:

* **Model Degradation**: Detect when model performance falls below acceptable thresholds
* **Resource Exhaustion**: Alert before running out of memory or disk space during training
* **API Health**: Identify when prediction services become unresponsive or error rates increase
* **Training Progress**: Alert on stalled training jobs or unexpected behavior
* **Data Issues**: Detect unusual patterns in input data that could indicate data drift

Proactive alerting allows you to address issues before they impact your users or corrupt training runs.

#### Set Up Alert Rules in Prometheus

Create a file named `prometheus/rules/ml_alerts.yml`:

```yaml
groups:
  - name: ml_application_alerts
    rules:
      # Alert for high prediction API error rate
      - alert: HighErrorRate
        expr: sum(rate(model_prediction_requests_total{status="error"}[5m])) / sum(rate(model_prediction_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate (> 10%)"
          description: "Prediction API error rate is {{ $value | humanizePercentage }} over the last 5 minutes."
          
      # Alert for slow prediction response time
      - alert: SlowPredictionResponse
        expr: histogram_quantile(0.95, sum(rate(model_prediction_duration_seconds_bucket[5m])) by (le, model_version)) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow prediction response time"
          description: "95th percentile response time for model {{ $labels.model_version }} is {{ $value }}s (>1s threshold)."
          
      # Alert for high memory usage
      - alert: HighMemoryUsage
        expr: app_memory_usage_bytes / 1024 / 1024 / 1024 > 1.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}GB, exceeding the 1.5GB threshold."
          
      # Alert for stalled training (no updates to metrics in 15 minutes)
      - alert: StalledTraining
        expr: time() - max(training_epochs_total_created) > 900
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training appears to be stalled"
          description: "No updates to training metrics in the last 15 minutes."
```

#### Update Prometheus Configuration

Modify your `prometheus.yml` to include the alert rules file:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Load alert rules
rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  # Your existing scrape configs...
```

#### Update Docker Compose

Update your docker-compose.yml to include the alert rules volume mapping for Prometheus:

```yaml
prometheus:
  image: prom/prometheus:latest
  # ... other config ...
  volumes:
    - ./prometheus:/etc/prometheus
    - ./prometheus/rules:/etc/prometheus/rules
    - prometheus-data:/prometheus
```

#### Integrating Alerts with External Systems

For production environments, you can extend your alerting by adding Alertmanager to send notifications to systems like:

* Email
* Slack
* PagerDuty
* Microsoft Teams
* Webhook endpoints

While setting up Alertmanager is beyond the scope of this assignment, understanding that Prometheus alerts can trigger external notifications is important for production deployments.

### 7. Run Your Containerized Application with Monitoring

Build and run your application with monitoring enabled:

```bash
docker-compose up --build
```

Access your monitoring tools:
* Grafana: http://localhost:3000 (login with admin/password or which ever you have set)
* Prometheus: http://localhost:9090

## Understanding Prometheus Query Language (PromQL)

PromQL is Prometheus's query language for selecting and aggregating time series data. Here are some basic examples:

### Simple Queries

```bash
# Get current training loss
training_loss

# Get rate of prediction requests over 5 minutes
rate(prediction_requests_total[5m])

# Get average prediction latency over last 5 minutes
rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])
```

### Aggregation Operators

```bash
# Sum of prediction requests by status
sum by(status) (prediction_requests_total)

# Average CPU usage over time
avg_over_time(training_cpu_usage_percent[10m])

# 90th percentile of prediction latency
histogram_quantile(0.9, sum(rate(prediction_duration_seconds_bucket[5m])) by (le, model_version))
```

### Best Practices for PromQL

1. **Start simple**: Build queries incrementally to understand each component
2. **Use rate() for counters**: Always use rate() or increase() when querying counters
3. **Label filtering**: Use label matchers to filter specific series (`{job="ml-api", status="error"}`)
4. **Avoid high cardinality**: Limit the number of unique label combinations
5. **Use recording rules**: For complex queries that are used frequently

## Best Practices for Prometheus Configuration

1. **Scrape interval**: Set appropriate scrape intervals based on metric volatility
   * High-frequency metrics: 5-15 seconds
   * Stable system metrics: 30-60 seconds

2. **Target labels**: Use meaningful job names and instance labels
   * `job` label identifies the type of service (e.g., "ml-api", "model-training")
   * `instance` label identifies specific instances

3. **Retention policy**: Configure storage based on your needs
   * Default is 15 days
   * Adjust with `--storage.tsdb.retention.time` flag

4. **Security**: Never expose Prometheus directly to the internet
   * Use reverse proxies
   * Implement authentication

## Best Practices for Grafana Dashboards

1. **Organization**:
   * Group related panels together
   * Use row dividers to separate logical sections
   * Name dashboards and panels meaningfully

2. **Visualization Selection**:
   * Time series graphs for trends over time
   * Gauges for current values against thresholds
   * Heatmaps for distribution metrics
   * Tables for detailed multi-dimensional data

3. **Consistency**:
   * Use consistent colors and formatting
   * Standardize units and decimal precision
   * Apply consistent time ranges

4. **Performance**:
   * Limit the number of panels per dashboard
   * Use template variables for filtering
   * Optimize complex queries with recording rules

5. **Alerting**:
   * Set up alerts for critical metrics
   * Include runbooks in alert notifications
   * Avoid alert fatigue with appropriate thresholds

## Additional Resources

### Prometheus
* [Official Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
* [PromQL Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
* [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
* [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)

### Grafana
* [Official Grafana Documentation](https://grafana.com/docs/grafana/latest/)
* [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
* [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/)
* [Grafana University](https://grafana.com/tutorials/) - Free online courses

## Submission Guidelines

Submit the following:

1. Updated project with Prometheus and Grafana integration (link to your repo)
2. Screenshots of your dashboards showing:
   * Model training metrics
   * Prediction API performance
   * Resource utilization during training (as many of you don't have an iterative training, showing training metrics at a given time is enough)
   * Alerts trigger by stressing your system (e.g. by sending requests to your API). This can be for a single alert rule.
3. A brief report explaining:
   * Your monitoring strategy
   * Key metrics you've chosen to track and why
   * How this monitoring would help detect issues in your ML pipeline
   * Any alerts you've configured

You should submit a single PDF file with all of the above. Make sure to include all the necessary details and screenshots.

**IMPORTANT NOTE**: I will still need to see collaboration in your repo in your commit history (if you are working in a single master repo) or you should have your own branch with your own commits if you are working in different branches for each team member.

## Evaluation Criteria

Your assignment will be evaluated based on:

1. Correct implementation of Prometheus metrics in your application
2. Proper configuration of Prometheus and Grafana
3. Quality and usefulness of the dashboards created
4. Understanding of monitoring concepts demonstrated in your report
5. Adherence to best practices for monitoring ML systems

## Hints and Tips

* Start with a few essential metrics rather than trying to monitor everything at once
* Test your monitoring with small training runs before full-scale training
* Use meaningful metric names and labels to make querying easier
* Explore Prometheus's alerting capabilities for proactive notification of issues
