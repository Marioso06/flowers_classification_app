# 🌸 Flower Classification – Makefile Guide

This project uses a `Makefile` to streamline environment setup and common tasks for training and running a flower classification model. Depending on your system’s hardware—AMD GPU, NVIDIA GPU, or CPU—you can initialize the environment accordingly and get started quickly.

---

## 🛠️ Makefile Commands

### 🔧 Environment Setup

Choose the appropriate command based on your hardware:

- **AMD GPU (ROCm)**  
  Sets up a virtual environment with PyTorch configured for AMD GPUs using ROCm.
  ```bash
  make init-amd
  ```

- **NVIDIA GPU (CUDA)**  
  Sets up a virtual environment with PyTorch configured for NVIDIA GPUs.
  ```bash
  make init-gpu
  ```

- **CPU Only**  
  Sets up a virtual environment with the CPU version of PyTorch.
  ```bash
  make init-cpu
  ```

---

### 🚀 Running Tasks

- **Train the Model**  
  Runs `train.py` from the `src/` directory to train your flower classification model.
  ```bash
  make train
  ```

- **Make Predictions**  
  Runs `predict.py` from the `src/` directory to generate predictions using the trained model.
  ```bash
  make predict
  ```

---

### 🧹 Cleaning Up

- **Clean Environment and Caches**  
  Removes `__pycache__` directories, virtual environment files, and pytest caches.
  ```bash
  make clean
  ```

---

### 📋 Show Help

- **List All Available Commands**  
  Displays a summary of all available Makefile targets.
  ```bash
  make help
  ```

---

## ⚙️ Environment Setup Instructions

1. **Select your hardware** and run the appropriate setup command:  
   - AMD GPU: `make init-amd`  
   - NVIDIA GPU: `make init-gpu`  
   - CPU: `make init-cpu`

2. **Activate the virtual environment:**
   ```bash
   source .flower_classification/bin/activate
   ```

3. **Install extra dependencies** (if needed) by editing the `requirements.txt` file.

---

## 📝 Notes

- Ensure your system supports the selected configuration:
  - ROCm for AMD GPUs
  - CUDA for NVIDIA GPUs

- The `init-amd` target includes additional ROCm-specific configurations.

- The `make clean` command deletes the virtual environment. Use with caution if you plan to reuse the environment.
