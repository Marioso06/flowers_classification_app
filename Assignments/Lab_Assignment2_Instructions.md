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

## NOTE

You should add `dvc`, `dvc-gdrive` and `mlflow` as dependencies in your `requirements.text` file. Remember that you can delete your python environement and create a new one or just run `pip3 install -r requirements.txt` within the root folder of your project. But remember that requirements does not include `Pytorch`, this dependency is always installed using the `make` command (one of the options like `make init-cpu`, `init-gpu etc`).
--
Notice the presence of two distinct YAML configuration files: `train_config.yaml` and `predict_config.yaml`. These files serve as centralized locations for setting parameters related to training and prediction, respectively. By using configuration files, you can adjust your experimental settings without altering the core codebase. This approach offers several benefits:

* Modularity: Keeping configuration separate from code makes your project cleaner and more organized.
* Flexibility: Easily change parameters (e.g., learning rates, batch sizes, model paths) by modifying the configuration files without risking unintended code changes.
* Reproducibility: Tracking changes in configuration files simplifies experiment replication and debugging.
* Collaboration: Team members can adjust settings independently, reducing merge conflicts and enhancing collaborative efforts.

For example, the `parameters.yml` file in the Flowers Classification repository on my GitHub serves a similar purpose by encapsulating all key settings and hyperparameters. This best practice is widely used in software development to maintain clean, maintainable, and scalable code.

Configuration files can also help manage sensitive information such as API keys, passwords, or tokens more securely by:

* Separation of Concerns: By isolating sensitive data in dedicated configuration files, you can prevent accidental exposure in your source code.
* Version Control Practices: Exclude configuration files that contain secrets from version control (e.g., using `.gitignore`) to ensure that sensitive information is not shared publicly.
* Access Control: Centralizing sensitive data in configuration files allows you to implement stricter access controls, ensuring that only authorized personnel or systems can access critical information.

For more information on how the `paramters.yml` in the Flowers Classification app works see the `How_arg_parser_works.md`

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

    Additionally, you can add any other data sources you might have (e.g, external)

3. **Using `.dvcignore`:**
    
    If there are subdirectories or files in `data/` that you do not want to track (e.g., temporary files), create or update the `.dvcignore` file in your project root. For example:

    ```bash
    #within your .dvcignore file you will add
    *.tmp
    cache/
    ```

    Expected Outcome: DVC creates `.dvc` files (e.g., `data/raw.dvc`, `data/processed.dvc`) that track the metadata of your datasets.

3. **Configure Remote Storage and Commit Changes**

    You will need to follow the instructions found here: [Using a custom Google Cloud project](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

    DON'T FORGET TO COMMIT YOUR CHANGES AND ALWAYS PRESS CRTL + S TO SAVE YOUR CHANGES!


# MLflow Setup and Integration Instructions

1. **Example of modifying an sckit-learn train.py code**

    You need to edit your training script to include MLflow logging. Below is an example modification:

    ```python
    # src/train.py
    import mlflow
    import mlflow.sklearn
    import yaml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib  # for saving your model

    # Load training configuration from YAML file
    with open('../configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Optionally, use MLflow autolog for sklearn
    mlflow.sklearn.autolog()

    # Start an MLflow run using the context manager
    with mlflow.start_run() as run:
        # Log training parameters
        mlflow.log_params(config)
        
        # --- Data loading and preprocessing steps go here ---
        
        # Example: Train a model
        model = RandomForestClassifier(**config.get('model_params', {}))
        model.fit(X_train, y_train)  # assume X_train and y_train are defined
        
        # Evaluate the model
        predictions = model.predict(X_test)  # assume X_test and y_test are defined
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        
        # Log the trained model
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Optionally, save the model to a file for local use
        joblib.dump(model, "../models/model.pkl")
        
        print(f"Run ID: {run.info.run_id}")
    ```

    **Key Points**:

    * Parameter Logging: 
    
        All parameters from your YAML configuration are logged using `mlflow.log_params(config)`.

    * Metric Logging: 
    
        Accuracy and any other metrics (like loss or F1-score) should be logged. Consider logging at multiple stages if needed.
    
    * Model Logging: 
    
        The trained scikit-learn model is logged with `mlflow.sklearn.log_model()`.
    
    * Autolog: 
    
        Enabling `mlflow.sklearn.autolog()` automatically logs parameters, metrics, and the model.

    **Expected Outcome**: When you run the training script, an MLflow run is created that captures configuration parameters, training metrics, and the saved model.

3. **Modify `evaluate.py` and `predict.py` to Load the Logged Model**
    
    For evaluation and prediction scripts, load the logged model from MLflow. For example:

    ```python
    # src/evaluate.py
    import mlflow.sklearn

    # Replace <RUN_ID> with the actual run ID from the MLflow run
    model_uri = "runs:/<RUN_ID>/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Proceed with evaluation
    predictions = model.predict(X_test)  # Ensure X_test is defined
    # ... evaluation logic here ...
    ```

    **Expected Outcome**: 
    
    You can now load and use models logged with MLflow for evaluation or prediction.

4. **Experiment Tracking and Comparison**

    Running Multiple Experiments

    * Modify hyperparameters in configs/train_config.yaml (e.g., adjust model parameters, learning rates, or other configurations used in your ML project).
    * Run the training script multiple times with different hyperparameters.
    * Each run will be recorded in MLflow, allowing you to compare performance metrics. Remember that each run you perform will be unique, pleasse check the `train.py` in the Flower Classification App on how you can create a unique name for each run in your experiment. It should look something like this:
    ```python
    # Start an MLflow run to track the experiment
    with mlflow.start_run(run_name=f"classification_{in_arg.arch}_{in_arg.training_compute}") as run:
    ```
    Note that the name is constructed using some of the parameters used for that run, that will help you have unique names in MLFlow.
    
    Compare Experiments in the MLflow UI
    
    * Use the MLflow UI to compare different runs side by side.
    Evaluate based on key metrics (accuracy, loss, F1-score etc) and choose the best configuration.
    
    **Expected Outcome**: 
    
    Through MLflow, you gain insights into how different hyperparameter configurations impact model performance.

5. **Git Integration (Code Versioning)**

    Versioning Code and DVC Files

    * Git is used to version control your code (e.g., Python scripts, configuration files).
    * DVC is used to version control your data (e.g., large datasets stored in data/).

    You should provide the link to your GitHub repository, within the repository I should be eable to see the commits that you saved to your project. Remember that a best practice is have small atomic commits that modify specefic pieces of your code. This to make your tracked changes more understandable.

    **Expected Outcome**: 
    
    A well-versioned repository where code and data changes can be traced over time.

# Deliverables

For this lab assignment, you must submit the following in a zip file:

**Screenshots**:

    Provide screenshots of the MLflow UI showing the tracked experiments (include run details such as parameters, metrics, and the logged model).

**Git Repository**:

    Submit a link to your Git repository that includes all the code modifications (e.g., changes in train.py, evaluate.py, predict.py, and any DVC files like .dvcignore and data/*.dvc).

**DVC Remote Storage Link**:

    Provide a link or reference to the remote storage where your data is versioned and sotored (remember that you will be using your Google Drive as a remote storage).