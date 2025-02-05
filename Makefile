# Project Variables
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Default arguments (empty)
ARGS=

# Create virtual environment
init:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

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
