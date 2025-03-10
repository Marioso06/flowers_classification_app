# Lab Assignment 3: Deploying a ML model using Flask web framework through a REST API

## Overview

The lab assignment’s purpose is to guide students through the process of creating and deploying a RESTful API in Flask that serves machine learning models, all while reinforcing best practices in version control. This will help you:

* Structure and expose ML prediction functions via multiple API endpoints.
* Handle incoming JSON requests, validate data, and return predictions.
* Compare results from two different serialized models (showcasing versioning strategies for APIs).
* Implement supporting endpoints (health and home) to communicate API status and usage instructions.
* Collaborate on the codebase in GitHub, demonstrating clear commit history and teamwork.

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
## End Goal

Have a running Flask application that exposes:

1. Two predict endpoints: `/v1/predict` (loads one model) and `/v2/predict` (loads a different model).
2. Health endpoint: `/health_status` – to confirm that the API is running and operational.
3. Home endpoint: `/{your_project}_home` – to provide usage information about the API.

Your Flask application should accept a payload containing the features needed for your ML model to generate predictions. These features reflect the specific machine learning problem chosen by your group. The application should be able to handle both single and multiple data samples. You will create two endpoints — `/v1/predict` and `/v2/predict` — to demonstrate how different model versions can be served under separate API routes, showcasing a practical approach to model version control.

# Assignment Instructions

Create a new file named `predict_api.py`. Begin by copying the relevant logic from your existing in the `predict.py` file into this new file, then update it according to the following guidelines so that your prediction functionality can be served via a Flask-based REST API.

1.	**Create `predict_api.py`**

    * **Location**: Place this new file inside your `src` folder
    * **Basic Flask Setup**:
        * Import `Flask`, `request`, and `jsonify` from `flask`.
        * Create a `Flas` instance.
        * Run the app by checking `if __name__ == "__main__": ....`

2.	**Routes**

    From the lecture, you know now that we can make specific functions from a python module accessible by using the `@app.route` decorator as an REST API in Flask. For this step, you will be creating the following routes so that you can add specific functionality to your API.

    * **Home Endpoint** (`/{your_project}_home`):
        
        This will be the default endpoint of your API. You should add the following information into it:
        
        * Briefly describe what the API is for. This is based on the specific functionality of your machine learning project.
        * Show the valid JSON format expected by the `/v1/predict` and `/v2/predict` endpoints.

        Return this information in a response (can be plain text, HTML, or JSON).

    * **Health Endpoint** (`/helth_status`):

        Return a message that will indicate the user that the API is available and ready. You can format this messsage as you please but keep it concise.

    * **Predict Endpoint v1** (`/v1/predict1`):

        * Load one of your previously saved mnodels from the `models/` folder (e.g., `model_test_1.pkl`)
        * Accept JSON data via a `POST` request.
        * Extract the features from the request payload.
        * Pass those features to the loaded model’s `predict()` method.
        * Return the predictions in a JSON response.

    * **Predict Endpoint v2** (`/v2/predict1`):

        * Load a different model from the `models/` folder (e.g., `model_test_2.pkl`).
        * Use the same approach as /v1/predict: read JSON payload, make a prediction, return it.
        * Show how two different models can be served in the same API.

3. **Handle the User Payload**
    
    In Flask, the `request` object gives you access to all of the data sent by the client in an HTTP request, including query parameters, form data, JSON payloads, and more. For this assignment we will focused on JSON like payloads.

    1. **Import the `request` Object**

        ```python
        from flask import Flask, request
        ```

    2. **Getting JSON Payloads**

        If the client sends JSON (usually with `Content-Type: application/json`), use `request.get_json()` to parse it into a Python dictionary or list:

        ```python
        data = request.get_json()
        # data is now a Python dictionary or list of dictionaries
        ```

    3. **Retrieving the entire raw request body**

        If needed, you can directly read the request body as bytes via `request.get_data()`:

        ```python
        raw_data = request.get_data()
        ```

    You need to make sure you handle the following requirements:

    * Check that all required fields are present and of the correct type.
    * Return the correct status codes (e.g., 400 Bad Request) if the necessary data is missing or malformed.

4. **Using the Models Folder in your Code**

    Make sure you are specifying how you will be accessing the models saved in your `models/` directory. This could be done as follows:

    ```python
    import pickle
    import os

    model_v1_path = os.path.join("models", "model_v1.pkl")
    model_v2_path = os.path.join("models", "model_v2.pkl")

    with open(model_v1_path, "rb") as f:
        model_v1 = pickle.load(f)

    with open(model_v2_path, "rb") as f:
        model_v2 = pickle.load(f)
    ```

    The models should be your **TOP TWO BEST** models.

    **IMPORTANT NOTE** 
    
    If your models are too big for GitHub, your should modify the code above so that you can import them using an URL (this can be done using Google Drive or OneDrive)

5. **Returning Responses from your Endpoints**

    In Flask, `jsonify` is a convenience function used to send JSON responses back to the client. It effectively wraps your Python data structures (like dictionaries or lists) into a valid JSON string and sets the appropriate response headers. Below is a detailed look at how jsonify works and why it is commonly used:

    - When you call `jsonify`, it uses Python's built-in `json` library to serialize your Python objects into a JSON-formatted string.

        ```python
        from flask import Flask, jsonify

        app = Flask(__name__)

        @app.route('/example')
        def example():
            data = {"message": "Hello, world!", "status": "success"}
            return jsonify(data)
        ```
        Internally, `jsonify(data)` is roughly doing:
        ```python
        reponse_body = json.dumps(data)
        ```

    - A primary benefit of jsonify is that it automatically sets the `Content-Type` header to `application/json`. This tells the client that the response is in JSON format, ensuring proper parsing on the client side.
    - Under the hood, `jsonify` returns a Flask `Response` object, populated with JSON data and headers. This `Response` object is what Flask sends to the client.
    - `jsonify` ensures that the response is UTF-8 encoded (the standard for JSON), minimizing compatibility issues with various clients.

    In general, Flaks' `jsonify` is great as it simplifies sending JSON data. It also ensure that correct geaders are set to avoid formatting erros. Furthermore, it is the best option for API development as APIs typically communicate using JSON. But at the end the best its best stregth is that maintains consistency in your application's behaviour.

    For this assignment you will use `jsonify` to return data in JSON form and to show HTTP status codes if something goes wrong, like so:

    ```python
    return jsonify({"prediction": str(prediction_result)})
    ```

# Documenting your API

You should create a detailed Markdown (`.md`) file that describes all the functionality of your API. Clearly outline the endpoints, required parameters, expected request body formats, possible responses, and any relevant usage examples. This documentation should specify what your API can accomplish, how to interact with it, and what clients or other developers should expect in terms of input/output. This to remember when building APIs:

* Clear documentation allows other developers, teams, or external services to quickly understand how to consume your API. This minimizes guesswork and speeds up integration.
* When endpoints, parameters, and data formats are explicitly defined, users of your API are less likely to make errors. This, in turn, decreases support overhead and troubleshooting time.
* New team members or collaborators can easily get up to speed on how to use or extend the API. Well-organized documentation provides a single source of truth, reducing the need for repetitive explanations.
* As your application grows or new versions of the API are released, well-structured documentation helps ensure consistency. It also provides a roadmap for future enhancements without introducing confusion or breaking changes.
* Developers can reference the documentation when writing unit or integration tests, ensuring coverage across all endpoints and helping them spot potential issues early in the development cycle.


# GitHub Version Control

**Collaboration & Commits**:

This is a group assignment, so you should see evidence of multiple contributors in the commit history. Encourage your group members branch for each feature, then merge back into the main branch with clear commit messages.

Example commits:

* `git commit -m "Add v1 predict endpoint"`
* `git commit -m "Fix bug when parsing JSON payload"`
* `git commit -m "Add docstrings to clarify prediction logic"`

**Final GitHub repository must reflect**:

1. The newly added `predict_api.py`.
2. Changes or updates to the `requirements.txt` to list Flask (if not already there).
3. Documentation describing each endpoint and usage.
4. A clean commit history showing how the code evolved.

# Deliverables

Through moodle you should only submit the link to your GitHub project repository alongside your team group number. No files should be submitted in Moodle!

**Important Note** if you submit your files in Moodle you will receive 0 automatically. Also, if you do not make the submission of your GitHub project repository you will get 0. Everyone needs to make a submission!!! **THERE ARE NO EXCEPTIONS**

Things I want to see in your GitHub:

1. `predict_api.py` module:
    * Must run via `python predict_api.py`
    * Exposes all the required endpoints

2. Two Predict Endpoints:
    * `/v1/predict` (e.g., loading `model_v1.pkl`)
    * `/v2/predict` (e.g., loading `model_v2.pkl`)

3. Health Endpoint:
    * `/health_status` returns information if the API is alive

4. Home Endpoint:
    * `/{your_project}_home` describing how the API works and the valid request payload.

5. Documentation:
    * Must clearly outline how to install dependencies, run the Flask app, and make requests.

6. GitHub History:
    * I must see in your GitHub project a series of commits that show collaboration and incremental development.

**Important**: If the code does not run or there are critical errors in the basic functionality, 40% of the mark will be deducted on the spot. Make sure your endpoints are callable and that the API returns appropriate responses.