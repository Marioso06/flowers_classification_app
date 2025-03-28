# Assignment: Converting Your ML Experiment into a Production-Ready Project

## Overview

In the CMPT3830 project (Alberta Food Drive), you were each assigned to a unique machine learning challenge depending on your group. Over the semester, you assembled a comprehensive Jupyter Notebook that included exploratory data analysis (EDA), data preprocessing, feature engineering, and model development—culminating in the validation of your chosen approaches. The goal was to harness relevant data to address real-world questions, such as how donations change year over year (Group 1), which neighborhoods might yield the highest or lowest donation volumes through geospatial analysis (Group 2), or how to optimize volunteer allocation for maximum route efficiency (Group 3). Some groups tackled ward-specific performance modeling to predict route efficiency (Group 4), while others investigated property assessment values and their impact on donation volumes (Group 5), or time-to-completion and donation prediction for improved scheduling insights (Group 6). There was even a sentiment analysis component (Group 7), which used volunteer comments to classify sentiments (positive, neutral, or negative) and explore correlations with donation success.

Despite the diversity in focus—be it Comparative Analysis, Geospatial Donation Prediction, Volunteer Efficiency Modeling, or Sentiment Classification—all these Jupyter Notebook experiments share a common need to be reproducible, maintainable, and ready for deployment. In other words, while the notebook stage is perfect for prototyping and rapid iteration, production projects demand best practices for code organization, versioning, testing, and environment management. By converting your existing work into a well-structured machine learning project, you ensure that others (or future versions of you!) can easily understand, reproduce, and extend the solution.

This next phase of your project asks you to apply software engineering and MLOps principles—modularizing your code, automating pipelines, and setting up environments that can be deployed across different systems. In doing so, you will transform your group’s specialized focus, whether it was Year-on-Year Comparison or Sentiment Analysis of Comments, into an industrial-strength project ready for real-world usage.


# Learning Objectives

1.	**Project Structuring**: Apply best practices to organize your project files and folders.
2.	**Software Engineering Principles**: Incorporate OOP patterns and well-documented, modularized code.
3.	**MLOps Understanding**: Practice the principles of the machine learning lifecycle, from data ingestion to model evaluation, and set up a project structure ready for continuous integration and deployment.
4.	**Configuration and Extensibility**: Demonstrate how to use configuration files to manage different training or preprocessing parameters.
5.  **Notebook Organization**: Create clear, focused Jupyter notebooks for exploration, prototyping, or demonstration, ensuring each notebook covers a specific task rather than cluttering a single notebook with all experimentation.

# Instructions

1.	**Identify Steps From Your Experiment**

    Review your initial Jupyter Notebook thoroughly. Identify all major steps you performed, such as:
        - Data loading
        - Data cleaning and preprocessing
        - Feature engineering
        - Model training (including hyperparameter tuning)
        - Model evaluation and performance metrics
        - Data visualization (if relevant for your final presentation)
        - Any custom utility functions

2.	**Map Experimental Steps to the Project Structure**

    Refer to the target folder structure below. Place your code accordingly. You may not use every folder/subfolder, but strive to incorporate as many of these structural components as make sense for your project.


    Project Structure

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
    └── .gitignore           # Not required for now
    ```

    * `data/`: Include raw data, processed data, or any external data sources.
    * `models/`: Save trained models (exported in your preferred format).
    * `notebooks/`: Keep your original Jupyter/Colab notebooks for reference or additional demos.
    * `src/`: Write scripts for data preprocessing, training, evaluation, and other helper functions.
    * `configs/`: Store YAML or JSON configuration files for preprocessing parameters, training hyperparameters, or environment-specific settings.
    * `logs/`: Capture logs or run outputs (optional at this stage, but beneficial to organize logs).
    * `experiments/`: Keep records or notes from your experiments, including model versions or logs from training runs.
    * `docs/`: Use for documentation (e.g., usage instructions, developer notes, architecture diagrams).
    * `requirements.txt` or environment.yml: Document all dependencies needed to run your project.
    * `Makefile`: Automate your setup steps (e.g., environment creation, running tests, formatting code).

3. **Create a `predict.py` Module (Optional but Highly Recommended)**

    Show how your trained model can be used independently by creating a module that accepts new data and outputs predictions. This can live in the `src/` folder or a subfolder like `src/prediction/`.

    A simple example of `predict.py` might look like this:

    ```python
    # src/predict.py

    import joblib
    import numpy as np

    class ModelPredictor:
        def __init__(self, model_path):
            """
            Initialize the predictor with a path to a trained model file.
            :param model_path: str, path to the .pkl or .joblib model file.
            """
            self.model = joblib.load(model_path)

        def predict(self, input_data: np.ndarray):
            """
            Make a prediction using the loaded model.
            :param input_data: np.ndarray containing the features for prediction.
            :return: np.ndarray of model predictions.
            """
            # Input validation or preprocessing can be handled here
            predictions = self.model.predict(input_data)
            return predictions

    if __name__ == "__main__":
        # Example usage:
        # 1. Instantiate the predictor
        predictor = ModelPredictor(model_path="../models/my_trained_model.joblib")

        # 2. Create some sample input data
        sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example shape for an Iris model

        # 3. Get predictions
        preds = predictor.predict(sample_input)
        print("Predictions:", preds)
    ```
    **Key Points:**

    * ModelPredictor is an independent software component. It does not require the rest of your project to run for predictions.
    * You can extend this class to handle model versioning, input validation, or pre-processing steps.

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