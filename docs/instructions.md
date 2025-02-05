# Building the Command-Line Application

Now that you've built and trained a deep neural network to classify flowers, it's time to turn it into an application that others can use. The application will consist of two Python scripts designed to run directly from the command line. These scripts should leverage the trained model and provide functionality for training and predicting flower classes. This activity expands on the initial instructions by integrating best practices for structuring machine learning projects. By the end, you will have a reusable, well-documented, and maintainable application.

For testing, you should use the checkpoint file that you saved earlier in the project.

---
# Project Structure

```bash
ml_project/
├── data/                # Stores datasets (raw, processed, external)
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/              # Saved or checkpointed model files (e.g., .pt, .pkl, .joblib)
├── notebooks/           # Jupyter/Colab notebooks for exploration/demos
├── src/
│   ├── train.py         # Script to train your model (CLI)
│   ├── predict.py       # Script to make predictions (CLI)
│   ├── preprocess.py    # Additional data preprocessing logic
│   ├── evaluate.py      # Model evaluation script
│   └── utils/           # Shared helper functions/classes
│       ├── model_utils.py
│       └── helpers.py
├── configs/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── tests/               # Tests for your scripts and utilities (optional)
├── logs/                # Training/evaluation logs (optional)
├── experiments/         # Experiment tracking (optional)
├── docs/                # Documentation (README, usage guides, etc.)
├── requirements.txt
├── environment.yml      # (optional)
├── Makefile
└── .gitignore           # Not required for now
```

**Key Actions**
1. Place your command-line scripts (`train.py`, `predict.py`) inside src/.
2. Organize additional support code (model loading, saving, argument parsing, and data utilities) within `src/utils/` (e.g., model_utils.py, helpers.py).
3. Store your trained model checkpoint in the `models/` directory for later use by `predict.py`.
4. Use configuration files (`.yaml` or `.json`) in `configs/` for hyperparameters, architecture selection, or path settings.
5. Include any relevant testing scripts (e.g., unit tests for argument parsing, integration tests for your training/prediction scripts) in `tests/`.

# Application Specifications

You will create two main CLI scripts:

## 1. **`train.py`** 

Trains a new neural network on a dataset and saves the model as a checkpoint file.
* **Basic Usage**:
    ```bash
    cd ml_project/src
    python train.py data_directory
    ```
    `data_directory` should point to the folder where your data resides (e.g., ../data/raw or another location).

Required Outputs:

- Print **training loss**, **validation loss**, and **validation accuracy** while training.
- Save a checkpoint (e.g., `.pt` file) in `ml_project/models/`.

**CLI Options** (via `argparse`):

1. `--save_dir`: Where to save the trained model checkpoint (default: `../models`)`.
    
    ```bash
    python train.py data_dir --save_dir ../models
    ```

2. `--arch`: Choose a **model architecture** (e.g., `vgg13`, `resnet50`)

    ```bash
    python train.py data_dir --arch "vgg13"
    ```

3. `--learning_rate`, `--hidden_units`, `--epochs`: Set hyperparameters for training.

    ```bash
    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    ```

4. `--gpu`: Use GPU for training if available.

    ```bash
    python train.py data_dir --gpu
    ```

**Implementation Tip:**
Store default hyperparameters and architecture names in configs/train_config.yaml. In train.py, load these defaults if the user does not provide explicit command-line values.


## 2. **`predict.py`**:

Uses the trained model checkpoint to classify a flower image.

* **Basic Usage**:
    ```bash
    cd ml_project/src
    python predict.py /path/to/image checkpoint
    ```
    The script outputs the predicted flower class (or classes) along with probabilities.

**CLI Options** (via `argparse`):

1. `--top_k`: Number of top likely classes to return (default: 1).
    
    ```bash
    python predict.py input checkpoint --top_k 3
    ```

2. `--category_names`: A JSON or YAML file mapping class indices to actual names.

    ```bash
    python predict.py input checkpoint --category_names ../data/cat_to_name.json
    ```

4. `--gpu`: Use GPU for training if available.

    ```bash
    python predict.py input checkpoint --gpu
    ```

**Implementation Tip:**
Use a helper function (in `src/utils/model_utils.py`) to:

* Load the checkpoint.
* Reconstruct the model architecture.
* Move the model to GPU/CPU as requested.

---

## Supportive Code Organization
While `train.py` and `predict.py` are required, you can (and should) organize your code by creating additional files:

1. **Abstract Model-Related Tasks**:

    * `model_utils.py`: Helper classes or functions for building, loading, and saving models.
    * `helpers.py`: Utility functions for image preprocessing, data loading, or common tasks (e.g., reading config files).

2. **Modularize**: Keep each script (train.py, predict.py) concise by delegating repeated tasks to your utility files.

## Error Handling, Testing and Aurgument Parsing

1. **Command-Line Argument Parsing**:
   Use the `argparse` module from Python’s standard library to handle command-line inputs. This is the most efficient and structured way to handle script arguments.  
   - You can find the official documentation for `argparse` [here](https://docs.python.org/3/library/argparse.html).
   - A helpful tutorial for beginners is available [here](https://realpython.com/command-line-interfaces-python-argparse/).

2. **Error Handling**:
   Add error handling to ensure your scripts fail gracefully if:
   - Required arguments are missing.
   - The checkpoint or image file is invalid or missing.
   - Unsupported architectures or hyperparameter values are provided.

3. **Testing**:
   Before submission, test both `train.py` and `predict.py` on various datasets, checkpoints, and configurations. Ensure the results are reproducible and align with expectations.

4. **Organized Output**:
   Make your output readable and informative, especially for command-line users. For example, when predicting, print both the class names and probabilities in a clear format.

## Documenetation

Provide clear instructions for:

1. **Environment Setup**:

    * List dependencies in requirements.txt or define an environment in environment.yml.
    * In a docs/ folder or a README.md, explain how to install and activate the environment (use a Makefile for automation).

2. **Running the Scripts**:

    * Document usage examples for train.py and predict.py.
    * Explain how to pass optional arguments (learning rate, number of epochs, GPU usage, etc.).
    * Show how to interpret the model’s output.

3. **Project Purpose**:

    * Provide a brief overview of what your project does (flower classification) and how it can be extended (e.g., for other image classification tasks).