# Project Variables
VENV=.flower_classification
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Default arguments (empty)
ARGS=

# Create virtual environment this is for AMD GPUs
init-amd-wsl:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	
	@echo "Intalling Pytorch for ROCm"
	wget -P /tmp https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl
	wget -P /tmp https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl
	wget -P /tmp https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl

	@echo "Uninstalling old versions of Pytorch..."
	$(PIP) uninstall -y torch torchvision pytorch-triton-rocm

	@echo "Installing downloaded ROCm packages..."
	$(PIP) install /tmp/torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl /tmp/torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl /tmp/pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl

	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt

	@echo "Finishing up setting up ROCm for Pytorch..."

	@location=$$($(PIP) show torch | grep Location | awk -F ": " '{print $$2}') && \
	echo "Torch installed at: $$location" && \
	test -n "$$location" || { echo "Error: Torch location is empty!"; exit 1; } && \
	test -f /opt/rocm/lib/libhsa-runtime64.so.1.2 || { echo "libhsa-runtime64.so.1.2 not found!"; exit 1; } && \
	cd $$location/torch/lib/ && \
	rm -f libhsa-runtime64.so* && \
	cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so

	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

# Create virtual environment this is for AMD GPUs
init-amd-linux:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip

	@echo "Installing downloaded ROCm packages..."
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt

	@echo "Finishing up setting up ROCm for Pytorch..."

	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

#Create virtual enviroment this is for CPU - General usage
init-cpu:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)

	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip

	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt

	@echo "Installing Pytorch - CPU"
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

#Create virtual enviroment - this if for Nvidia GPU
init-gpu:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)

	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip

	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt

	@echo "Installing Pytorch - GPU Nvidia"
	$(PIP) torch torchvision torchaudio

	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

#Create virtual enviroment 
init-mac:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)

	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip

	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt

	@echo "Installing Pytorch - Mac"
	$(PIP) install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

# MLFlow Initialization
mlflow-init:
	@echo "Initializing MLFlow server"
	$(PYTHON) src/utils/mlflow_initialization.py

train:
	@echo "Running train.py..."
	$(PYTHON) src/train.py

# Run Tests
predict:
	@echo "Running tests..."
	$(PYTHON) src/predict.py

# Clean up cache & logs
clean:
	@echo "Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)
	rm -rf .pytest_cache

# Show available commands
help:
	@echo "Makefile commands:"
	@echo "  make init      - Create virtual env and install dependencies"
	@echo "  make install   - Install dependencies from requirements.txt"
	@echo "  make train     - Run the training script"
	@echo "  make predict   - Run the predict script"
	@echo "  make clean     - Remove cache files and virtual environment"
