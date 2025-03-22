# Flowers Classification App: AWS Deployment Guide

This guide provides step-by-step instructions for deploying the Flowers Classification App on Amazon Web Services (AWS). The deployment workflow follows these logical steps:

1. Setting up S3 storage and uploading data
2. Deploying MLflow with RDS PostgreSQL and S3 integration
3. Setting up an EC2 instance for model training
4. Training models and tracking experiments with MLflow
5. Deploying the prediction API to AWS Fargate

## Prerequisites

Before starting, ensure you have:

- An Amazon Web Services account ([sign up here](https://aws.amazon.com/free/) if needed)
- Basic familiarity with AWS and its console
- The [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- Access to the Flowers Classification App repository

## Getting Started with AWS Cloud9 (Optional)

For a cloud-based development environment, you can use AWS Cloud9:

1. Go to the [AWS Management Console](https://console.aws.amazon.com/)
2. Navigate to Cloud9 and create a new environment
3. Once your Cloud9 environment is ready, clone the repository:
   ```bash
   git clone https://github.com/your-repo/flowers_classification_app.git
   cd flowers_classification_app
   ```

## 1. Setting Up S3 Storage and Uploading Data

### Step 1: Create an S3 Bucket

```bash
# Set a unique bucket name
BUCKET_NAME="flowers-classification-data-$(date +%s)"

# Create the bucket
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Export as an environment variable for later use
export AWS_S3_BUCKET_NAME=$BUCKET_NAME
echo "Your bucket name is: $BUCKET_NAME"
```

### Step 2: Upload Training Data to the Bucket

First, we need to get the data into our environment by executing the following commands:

```bash
mkdir -p data/raw && cd data/raw
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=18I2XurHF94K072w4rM3uwVjwFpP_7Dnz&export=download&confirm=t&uuid=7feba52a-4578-499a-b3bc-469e687781f4" -O flower_data.tar.gz
tar -xzf flower_data.tar.gz
cd ../..

# Upload data to the bucket
aws s3 cp --recursive data/raw/flower_data s3://$BUCKET_NAME/data/

# Verify data was uploaded
aws s3 ls s3://$BUCKET_NAME/data/ --summarize
```

### Step 3: Verify the AWS Configuration

The app comes with an AWS configuration file that contains settings for bucket names, MLflow, and database configuration. Let's verify it:

```bash
# View the AWS configuration file content
cat configs/aws_config.py
```

Verify that the S3 utility functions are available in `src/utils/s3_utils.py`:

```bash
# Check that S3 utilities exist
cat src/utils/s3_utils.py
```

These utilities allow the application to download/upload files from/to Amazon S3.

## 2. Setting Up MLflow with RDS PostgreSQL and S3

### Step 1: Create an RDS PostgreSQL Instance

```bash
# Create a security group for the RDS instance
aws ec2 create-security-group --group-name mlflow-db-sg --description "Security group for MLflow database"

# Get your IP address to restrict access to the database
YOUR_IP=$(curl -s https://checkip.amazonaws.com)

# Add a rule to allow database connections from your IP
aws ec2 authorize-security-group-ingress \
    --group-name mlflow-db-sg \
    --protocol tcp \
    --port 5432 \
    --cidr $YOUR_IP/32

# Create a PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier mlflow-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username mlflow \
    --master-user-password "SECURE_PASSWORD_HERE" \
    --allocated-storage 20 \
    --vpc-security-group-ids $(aws ec2 describe-security-groups --group-names mlflow-db-sg --query "SecurityGroups[0].GroupId" --output text) \
    --db-name mlflow

# Wait for the database to be available
aws rds wait db-instance-available --db-instance-identifier mlflow-db

# Get the connection details
RDS_ENDPOINT=$(aws rds describe-db-instances --db-instance-identifier mlflow-db --query "DBInstances[0].Endpoint.Address" --output text)
echo "RDS Endpoint: $RDS_ENDPOINT"

# Store these securely
export DB_HOST=$RDS_ENDPOINT
export DB_USER="mlflow"
export DB_PASSWORD="SECURE_PASSWORD_HERE"
export DB_NAME="mlflow"
```

### Step 2: Create an EC2 Instance for MLflow

```bash
# Create a key pair for SSH access (if you don't have one)
aws ec2 create-key-pair --key-name mlflow-key --query "KeyMaterial" --output text > mlflow-key.pem
chmod 400 mlflow-key.pem

# Create a security group for the MLflow server
aws ec2 create-security-group --group-name mlflow-server-sg --description "Security group for MLflow server"

# Add rules to allow SSH and MLflow access
aws ec2 authorize-security-group-ingress \
    --group-name mlflow-server-sg \
    --protocol tcp \
    --port 22 \
    --cidr $YOUR_IP/32

aws ec2 authorize-security-group-ingress \
    --group-name mlflow-server-sg \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0

# Create an IAM role for EC2 to access S3
aws iam create-role --role-name MLflowEC2Role --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach S3 access policy to the role
aws iam attach-role-policy --role-name MLflowEC2Role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create an instance profile and associate it with the role
aws iam create-instance-profile --instance-profile-name MLflowEC2Profile
aws iam add-role-to-instance-profile --instance-profile-name MLflowEC2Profile --role-name MLflowEC2Role

# Launch an EC2 instance for MLflow
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t2.medium \
    --key-name mlflow-key \
    --security-group-ids $(aws ec2 describe-security-groups --group-names mlflow-server-sg --query "SecurityGroups[0].GroupId" --output text) \
    --iam-instance-profile Name=MLflowEC2Profile \
    --user-data "$(cat << 'EOF'
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x

# Update and install dependencies
apt-get update
apt-get install -y git python3-pip python3-venv postgresql-client

# Clone the repository
mkdir -p /opt/mlflow
cd /opt/mlflow
git clone https://github.com/Marioso06/flowers_classification_app.git
cd flowers_classification_app

# Set up Python environment
python3 -m venv .mlflow_env
source .mlflow_env/bin/activate
pip install --upgrade pip
pip install mlflow==2.20.2 boto3 psycopg2-binary

# Create a configuration file for running MLflow later
cat > /opt/mlflow/run_mlflow.sh << 'INNEREOF'
#!/bin/bash
cd /opt/mlflow/flowers_classification_app
source .mlflow_env/bin/activate

# Set environment variables
export AWS_S3_BUCKET_NAME="${AWS_S3_BUCKET_NAME}"
export USE_S3_FOR_MLFLOW="true"
export MLFLOW_HOST="0.0.0.0"
export MLFLOW_PORT="5000"
export DB_HOST="${DB_HOST}"
export DB_USER="${DB_USER}"
export DB_PASSWORD="${DB_PASSWORD}"
export DB_NAME="${DB_NAME}"
export MLFLOW_DB_URI="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}"
export USE_POSTGRES_FOR_MLFLOW="true"
export PYTHONPATH=/opt/mlflow/flowers_classification_app:$PYTHONPATH

# Start MLflow
nohup python src/utils/mlflow_initialization.py --host 0.0.0.0 --port 5000 --use-s3 --use-postgres > mlflow.log 2>&1 &
echo "MLflow server started on port 5000"
INNEREOF

chmod +x /opt/mlflow/run_mlflow.sh
echo "VM setup complete. Ready to start MLflow after RDS authorization."
EOF
)" \
    --query "Instances[0].InstanceId" \
    --output text)

# Wait for the instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get the public IP address of the instance
EC2_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "MLflow server EC2 IP: $EC2_PUBLIC_IP"
```

### Step 3: Authorize the EC2 Instance to Connect to RDS

We need to update the security group of our RDS instance to allow connections from the EC2 instance:

```bash
# Get the security group ID of the EC2 instance
EC2_SG_ID=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" --output text)

# Modify the RDS security group to allow access from the EC2 instance
aws ec2 authorize-security-group-ingress \
    --group-name mlflow-db-sg \
    --protocol tcp \
    --port 5432 \
    --source-group $EC2_SG_ID
```

### Step 4: Connect to the EC2 Instance and Start MLflow

```bash
# SSH into the instance
ssh -i mlflow-key.pem ubuntu@$EC2_PUBLIC_IP

# Once connected, update the environment variables in the run_mlflow.sh script
sudo sed -i "s/\${AWS_S3_BUCKET_NAME}/$AWS_S3_BUCKET_NAME/g" /opt/mlflow/run_mlflow.sh
sudo sed -i "s/\${DB_HOST}/$DB_HOST/g" /opt/mlflow/run_mlflow.sh
sudo sed -i "s/\${DB_USER}/$DB_USER/g" /opt/mlflow/run_mlflow.sh
sudo sed -i "s/\${DB_PASSWORD}/$DB_PASSWORD/g" /opt/mlflow/run_mlflow.sh
sudo sed -i "s/\${DB_NAME}/$DB_NAME/g" /opt/mlflow/run_mlflow.sh

# Start MLflow server
sudo /opt/mlflow/run_mlflow.sh

# Verify MLflow is running
curl http://localhost:5000
```

You can now access the MLflow UI by visiting `http://$EC2_PUBLIC_IP:5000` in your browser.

## 3. Setting Up an EC2 Instance for Model Training

### Step 1: Create an EC2 Instance for Training

```bash
# Create a security group for the training instance
aws ec2 create-security-group --group-name training-sg --description "Security group for model training"

# Add a rule to allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-name training-sg \
    --protocol tcp \
    --port 22 \
    --cidr $YOUR_IP/32

# Launch the training instance
TRAINING_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t2.large \
    --key-name mlflow-key \
    --security-group-ids $(aws ec2 describe-security-groups --group-names training-sg --query "SecurityGroups[0].GroupId" --output text) \
    --iam-instance-profile Name=MLflowEC2Profile \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":30,\"DeleteOnTermination\":true}}]" \
    --query "Instances[0].InstanceId" \
    --output text)

# Wait for the instance to be running
aws ec2 wait instance-running --instance-ids $TRAINING_INSTANCE_ID

# Get the public IP address of the instance
TRAINING_IP=$(aws ec2 describe-instances --instance-ids $TRAINING_INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Training instance IP: $TRAINING_IP"
```

### Step 2: Connect to the Training Instance and Set It Up

```bash
# SSH into the instance
ssh -i mlflow-key.pem ubuntu@$TRAINING_IP

# Once connected to the instance, set up the environment
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Clone the repository
git clone https://github.com/Marioso06/flowers_classification_app.git
cd flowers_classification_app

# Set up Python environment
python3 -m venv .flower_classification
source .flower_classification/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3

# Add AWS credentials (if not using instance profile)
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
EOF

# Set environment variables
export AWS_S3_BUCKET_NAME=$AWS_S3_BUCKET_NAME
export MLFLOW_TRACKING_URI="http://$EC2_PUBLIC_IP:5000"
export USE_S3_FOR_MLFLOW="true"
```

## 4. Training Models and Tracking Experiments with MLflow

### Step 1: Configure and Run Training

While connected to the training instance:

```bash
# Configure AWS settings
cd ~/flowers_classification_app
source .flower_classification/bin/activate

# Add these to your .bashrc for persistence
echo "export AWS_S3_BUCKET_NAME=$AWS_S3_BUCKET_NAME" >> ~/.bashrc
echo "export MLFLOW_TRACKING_URI=http://$EC2_PUBLIC_IP:5000" >> ~/.bashrc

# Run a basic training job with MLflow tracking
python src/train.py \
    --data_dir s3://$AWS_S3_BUCKET_NAME/data/flower_data \
    --save_dir models \
    --arch vgg13 \
    --learning_rate 0.001 \
    --hidden_units 512 \
    --epochs 3 \
    --gpu
```

### Step 2: Track Experiments in MLflow

Access the MLflow UI at `http://$EC2_PUBLIC_IP:5000` to monitor training progress and compare experiments.

## 5. Deploying the Prediction API to AWS Fargate

### Step 1: Create an ECR Repository

```bash
# Create a repository for the prediction API
aws ecr create-repository --repository-name flowers-classification-api

# Get the repository URI
ECR_REPO_URI=$(aws ecr describe-repositories --repository-names flowers-classification-api --query "repositories[0].repositoryUri" --output text)
echo "ECR Repository URI: $ECR_REPO_URI"
```

### Step 2: Build and Push Docker Image

Create a Dockerfile in your project directory if it doesn't exist:

```bash
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3 gunicorn

COPY . .

ENV AWS_S3_BUCKET_NAME=$AWS_S3_BUCKET_NAME
ENV MLFLOW_TRACKING_URI=http://$EC2_PUBLIC_IP:5000
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=9000

CMD ["gunicorn", "--bind", "0.0.0.0:9000", "predict_api:app"]
EOF
```

Build and push the Docker image:

```bash
# Log in to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO_URI

# Build the Docker image
docker build -t flowers-classification-api .

# Tag the image
docker tag flowers-classification-api:latest $ECR_REPO_URI:latest

# Push the image to ECR
docker push $ECR_REPO_URI:latest
```

### Step 3: Create an ECS Cluster and Task Definition

```bash
# Create an ECS cluster
aws ecs create-cluster --cluster-name flowers-cluster

# Create a task execution role
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach the necessary policies
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create a task definition
cat > task-definition.json << EOF
{
  "family": "flowers-api",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "flowers-api",
      "image": "$ECR_REPO_URI:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 9000,
          "hostPort": 9000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AWS_S3_BUCKET_NAME",
          "value": "$AWS_S3_BUCKET_NAME"
        },
        {
          "name": "MLFLOW_TRACKING_URI",
          "value": "http://$EC2_PUBLIC_IP:5000"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/flowers-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": [
    "FARGATE"
  ],
  "cpu": "1024",
  "memory": "2048"
}
EOF

# Register the task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Step 4: Create a Load Balancer and Security Group

```bash
# Create a security group for the load balancer
aws ec2 create-security-group --group-name flowers-lb-sg --description "Security group for flowers API load balancer"

# Allow HTTP traffic
aws ec2 authorize-security-group-ingress \
    --group-name flowers-lb-sg \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Get the default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

# Create a load balancer
aws elbv2 create-load-balancer \
    --name flowers-lb \
    --subnets ${SUBNET_IDS//,/ } \
    --security-groups $(aws ec2 describe-security-groups --group-names flowers-lb-sg --query "SecurityGroups[0].GroupId" --output text)

# Get the load balancer ARN
LB_ARN=$(aws elbv2 describe-load-balancers --names flowers-lb --query "LoadBalancers[0].LoadBalancerArn" --output text)

# Create a target group
aws elbv2 create-target-group \
    --name flowers-targets \
    --protocol HTTP \
    --port 9000 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-path "/health" \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 2

# Get the target group ARN
TG_ARN=$(aws elbv2 describe-target-groups --names flowers-targets --query "TargetGroups[0].TargetGroupArn" --output text)

# Create a listener
aws elbv2 create-listener \
    --load-balancer-arn $LB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TG_ARN
```

### Step 5: Create a Security Group for Fargate Tasks

```bash
# Create a security group for Fargate tasks
aws ec2 create-security-group --group-name flowers-task-sg --description "Security group for flowers API Fargate tasks"

# Allow traffic from the load balancer
aws ec2 authorize-security-group-ingress \
    --group-name flowers-task-sg \
    --protocol tcp \
    --port 9000 \
    --source-group $(aws ec2 describe-security-groups --group-names flowers-lb-sg --query "SecurityGroups[0].GroupId" --output text)
```

### Step 6: Create and Run the Fargate Service

```bash
# Create a CloudWatch log group
aws logs create-log-group --log-group-name /ecs/flowers-api

# Create the service
aws ecs create-service \
    --cluster flowers-cluster \
    --service-name flowers-api \
    --task-definition flowers-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS//,/ }],securityGroups=[$(aws ec2 describe-security-groups --group-names flowers-task-sg --query "SecurityGroups[0].GroupId" --output text)],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=flowers-api,containerPort=9000"

# Get the load balancer DNS name
LB_DNS=$(aws elbv2 describe-load-balancers --names flowers-lb --query "LoadBalancers[0].DNSName" --output text)
echo "Your API is accessible at: http://$LB_DNS"
```

## 6. Monitoring and Maintenance

### Monitoring with CloudWatch

```bash
# Create a dashboard for monitoring
aws cloudwatch put-dashboard \
    --dashboard-name FlowersApiDashboard \
    --dashboard-body "{\"widgets\":[{\"type\":\"metric\",\"x\":0,\"y\":0,\"width\":12,\"height\":6,\"properties\":{\"metrics\":[[\"AWS/ECS\",\"CPUUtilization\",\"ServiceName\",\"flowers-api\",\"ClusterName\",\"flowers-cluster\"]],\"period\":300,\"stat\":\"Average\",\"region\":\"us-east-1\",\"title\":\"CPU Utilization\"}}]}"
```

### Setting Up Alarms

```bash
# Create an alarm for high CPU usage
aws cloudwatch put-metric-alarm \
    --alarm-name flowers-api-high-cpu \
    --alarm-description "Alarm when CPU exceeds 80% for 5 minutes" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --dimensions Name=ClusterName,Value=flowers-cluster Name=ServiceName,Value=flowers-api \
    --statistic Average \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --period 300 \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:NotifyMe
```

### Setting Up AWS Secrets Manager (Alternative to Environment Variables)

```bash
# Create a secret for sensitive configuration
aws secretsmanager create-secret \
    --name flowers/mlflow-db-credentials \
    --secret-string "{\"username\":\"$DB_USER\",\"password\":\"$DB_PASSWORD\",\"host\":\"$DB_HOST\",\"port\":\"5432\",\"dbname\":\"$DB_NAME\"}"
```

## 7. Cleanup

When you're done with the resources, you can clean them up to avoid ongoing charges:

```bash
# Delete the Fargate service
aws ecs update-service --cluster flowers-cluster --service flowers-api --desired-count 0
aws ecs delete-service --cluster flowers-cluster --service flowers-api --force

# Delete the cluster
aws ecs delete-cluster --cluster flowers-cluster

# Delete the load balancer and target group
aws elbv2 delete-load-balancer --load-balancer-arn $LB_ARN
aws elbv2 delete-target-group --target-group-arn $TG_ARN

# Terminate EC2 instances
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 terminate-instances --instance-ids $TRAINING_INSTANCE_ID

# Delete the RDS instance
aws rds delete-db-instance --db-instance-identifier mlflow-db --skip-final-snapshot

# Delete security groups
aws ec2 delete-security-group --group-name mlflow-server-sg
aws ec2 delete-security-group --group-name training-sg
aws ec2 delete-security-group --group-name flowers-lb-sg
aws ec2 delete-security-group --group-name flowers-task-sg
aws ec2 delete-security-group --group-name mlflow-db-sg

# Empty and delete the S3 bucket
aws s3 rm s3://$AWS_S3_BUCKET_NAME --recursive
aws s3 rb s3://$AWS_S3_BUCKET_NAME

# Delete IAM roles
aws iam remove-role-from-instance-profile --instance-profile-name MLflowEC2Profile --role-name MLflowEC2Role
aws iam delete-instance-profile --instance-profile-name MLflowEC2Profile
aws iam detach-role-policy --role-name MLflowEC2Role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name MLflowEC2Role
aws iam detach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam detach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam delete-role --role-name ecsTaskExecutionRole
```

## Conclusion

This guide provides a comprehensive approach to deploying the Flowers Classification App on AWS with a focus on production-ready MLflow integration. By following these steps, you have:

1. Set up cloud storage for your data and models
2. Deployed MLflow with a PostgreSQL backend for reliable experiment tracking
3. Created a training environment for your models
4. Set up a containerized prediction API with Fargate for scalability

The architecture follows AWS best practices and leverages managed services where possible to minimize operational overhead. For production environments, consider adding additional security measures such as VPC endpoints, private subnets, and more restrictive security groups.
