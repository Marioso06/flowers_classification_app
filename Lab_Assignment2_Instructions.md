# Lab Assignment 2: Integrating DVC and MLflow for Machine Learning Deployment

## Overview

In this lab assignment, you will enhance your existing machine learning project by incorporating data versioning with DVC and experiment tracking with MLflow. You will learn how to:

* Initialize and use DVC for managing your data (both raw and processed)
* Integrate MLflow into your training script to log parameters, metrics, and models
* Compare multiple experiments using MLflow’s UI
* Use Git alongside DVC to ensure both code and data have complete version histories

Your project structure already looks like this:
```bash
ml_project/
├── data/                # Stores datasets (raw, processed, external)
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/              # Saved or checkpointed model files (e.g., .pt, .pkl, .joblib)
├── notebooks/           # Jupyter/Colab notebooks for exploration/demos
├── src/
│   ├── train.py         # Script to train your model
│   ├── predict.py       # Script to make predictions
│   ├── preprocess.py    # Additional data preprocessing logic
│   ├── evaluate.py      # Model evaluation script
│   └── utils/           # Shared helper functions/classes
│       ├── model_utils.py
│       └── helpers.py
├── configs/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── docs/                # Documentation (README, usage guides, etc.)
├── requirements.txt
├── Makefile
└── .gitignore
```

# DVC Instructions

1.	**DVC Setup and Data Versioning**

    * Open a terminal and navigate to the root of your project directory
    * Initialize DVC:
        ```bash
        dvc init
        ```
    Expected Outcome: A `.dvc/` directory is created and configuration files are updated.

2.	**Track Your Data Directory**

    **Track Raw Data**
    Since raw data is large and should remain immutable, add the raw data directory to DVC:
        ```bash
        dvc add data/raw
        ```
    
    **Track Processed  Data**
    Processed data might change as you update your pipeline, but it will always come from the same source and from the same processes (that should be version controlled as well). You can add the folder like so:
        ```bash
        dvc add data/processed
        ```
    
3. **Using `.dvcignore`:**
    
    If there are subdirectories or files in data/ that you do not want to track (e.g., temporary files), create or update the .dvcignore file in your project root. For example:

        ```bash
        #within your .dvcignore file you will add
        *.tmp
        cache/
        ```

    Expected Outcome: DVC creates .dvc files (e.g., `data/raw.dvc`, `data/processed.dvc`) that track the metadata of your datasets.

3. **Configure Remote Storage and Commit Changes**

    You will need to follow the instructions found here: [Using a custom Google Cloud project](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

    DON'T FORGET TO COMMIT YOUR CHANGES!!


# MLflow Setup and Integration Instructions

1. **Install MLflow**
    ```bash
    

4. **Write Your Training and Preprocessing Scripts**

    * Preprocessing (`preprocess.py`): Handle data cleaning, feature engineering, splitting into training/testing sets. Configure parameters such as missing value strategies or feature selection in configs/preprocess_config.yaml.
    * Training (`train.py`): Train your model, implement hyperparameter tuning if needed, and save the trained model to the `models/` folder. (Optional) You refer to `configs/train_config.yaml` for parameters (e.g., learning rate, epochs, or random seed).
    * Evaluation (`evaluate.py`): Evaluate your model’s performance using test data or hold-out datasets, and log or save metrics for later reference.

5. **Utilize Object-Oriented Programming**
    Where possible, encapsulate functionality into classes within your src/utils/helpers.py or other modules (like `preprocess.py`, `train.py` etc). For instance, you could create:

    * A DataLoader class for reading datasets from data/raw/.
    * A FeatureEngineer class for generating new features.
    * A Trainer class for running training jobs.
    * A Evaluator class for evaluating the trained model.
    
    These classes should be designed to plug and play within your scripts, so they can also be reused or extended in future projects.

6. **Notebook Organization**
    Place any Jupyter or Colab notebooks inside the `notebooks/` folder. Instead of having a single monolithic notebook, you could create multiple notebooks, each covering a specific aspect of your workflow—such as:

    * Exploratory Data Analysis
    * Feature Engineering / Data Cleaning
    * Model Prototyping / Comparisons
    * Visualization / Interpretations
    
    While this is not extrictly necessary, this approach could help you ensure that each notebook is focused and modular, making it easier for others (and your future self) to navigate and understand each step of the process. If you decide not to break your notebooks (which is fine) you are expected at least to have all your notebooks in the `notebooks/` folder with proper naming convention depending of the objective of that notebook.

7. **Document Your Project**

    * Add docstrings to all classes, functions, and modules.
    * Create or update the docs/ folder with a README.md or index.md explaining how to install dependencies, run preprocessing, train the model, evaluate the model, and make predictions.
    * Add comments describing any design decisions (e.g., “We chose this architecture because …”).

8. **`requirements.txt` and `makefile`**
    * Makefile: Include tasks for environment setup, testing, training, data acquisition etc. You can select just the enviroment setup if you don't have any automated process.
    * requirements.txt: Include a text file with the requirements of you project.


# Deliverables

A zip file that contains all of the following aspects:

1. Project Structure: A well-organized folder structure as described.
2. Modular Code: Demonstrate OOP and modular design in your scripts.
3. Configuration Files: YAML or JSON configs in the configs/ folder (Optinal).
4. Documentation: README or docs describing how to run your project.
5. Notebook Organization: Clear naming convention and organization of all your jupyter notebooks used in your CMPT3510 project


# Final Notes

This assignment sets the stage for you to practice real-world machine learning development workflows, where teamwork, versioning, containerization, and deployment matter just as much as model accuracy. By structuring your code in this manner, you make it easier to integrate with CI/CD pipelines, scale your work to more complex projects, and facilitate collaboration with other data scientists and engineers.

Feel free to tailor details (e.g., additional scripts or different file naming conventions) to suit your team’s project requirements, but keep in mind the core objectives: maintain clarity, modularity, and a structure that can scale for future needs.