# Flower Classification Makefile Guide

This project provides a Makefile to manage the environment setup and tasks for training and prediction of a flower classification model. Depending on your hardware setup (AMD GPU, NVIDIA GPU, or CPU), you can initialize the environment accordingly.

## Makefile Commands

### Environment Setup
- **AMD GPU Setup (ROCm)**

```bash
make init-amd
```

This command sets up a virtual environment with PyTorch optimized for AMD GPUs using ROCm.

- **CPU Setup**

```bash
make init-cpu
```

This command sets up a virtual environment with PyTorch for CPU-based processing.

- **NVIDIA GPU Setup**

```bash
make init-gpu
```
This command sets up a virtual environment with PyTorch optimized for NVIDIA GPUs.

### Running Tasks
- **Train the Model**

```bash
make train
```
Executes the `train.py` script located in the `src` directory to train the flower classification model.

- **Run Predictions**

```bash
make predict
```
Executes the `predict.py` script located in the `src` directory to make predictions with the trained model.

### Cleaning Up
- **Clean Cache and Logs**

```bash
make clean
```
Removes `__pycache__` directories, the virtual environment, and pytest cache files.

### Help
- **Display Available Commands**

```bash
make help
```
Lists all available Makefile commands with a brief description.

---

## Instructions for Environment Setup

1. Choose the appropriate setup based on your hardware:
 - **AMD GPU**: Use `make init-amd`.
 - **CPU**: Use `make init-cpu`.
 - **NVIDIA GPU**: Use `make init-gpu`.
 
2. After the setup is complete, activate the virtual environment:

```bash
source .flower_classification/bin/activate
```

3. Install additional dependencies if needed by editing `requirements.txt`.

---

## Notes
- Ensure your hardware supports the chosen setup (ROCm for AMD, CUDA for NVIDIA).
- For AMD GPU setup, specific ROCm libraries and configurations are handled in the `init-amd` target.
- The `clean` command will remove the virtual environment, so use it with caution if you need to keep the environment for future use.
