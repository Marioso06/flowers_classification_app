# How `arg_parser.py` Controls Parameters in `train.py`

**Loading Default Configurations from `parameters.yml`**:

The `arg_parser.py` script begins by loading default parameters from `parameters.yml`, which includes things like dataset paths, model architecture, training hyperparameters, etc. This means you don’t have to hard-code values in your `train.py`. Instead, you keep them in a central `parameters.yml` file.

**Command-Line Argument Parsing**:

Inside `arg_parser.py`, the Python argparse library is used to define command-line arguments that map to the defaults loaded from `parameters.yml`. For instance:

```python
parser.add_argument('--data_directory', type=str, default=config["data_directory"], ...)
parser.add_argument('--arch', type=str, default=config["arch"], ...)
```

This means that if you run:

```python
python src/train.py --data_directory /path/to/custom_data --arch vgg13
```

those command-line values will override the defaults from `parameters.yml`.

**Flexible Parameter Control in train.py**:

In your `train.py`, you import `get_input_args()` from `arg_parser.py`. When you call this function:

```python
in_arg = get_input_args()
```

you receive a namespace (in_arg) containing all the parameters specified in `parameters.yml`, updated by any command-line overrides. This makes it very easy to:

* Change hyperparameters without touching your code.
* Switch datasets or checkpoint directories via command-line flags.
* Keep a clean separation between code logic and configuration.

**Example Flow**:

When you run your training script, the data directory, architecture, learning rate, etc. are all pulled from either command-line arguments or the default YAML config. Here’s a simplified flow:

1. `train.py` calls `get_input_args()`.
2. `get_input_args()` loads parameters from `parameters.yml`.
3. `get_input_args()` also looks for command-line arguments.
4. The function merges them (command-line overrides config file).
5. `in_arg` is returned to `train.py`, where you use these values to configure your model, data loaders, and other components.

# Importance of a Central Configuration File (`parameters.yml`)

**Reproducibility**:
Having all default parameters in one YAML file makes it easy to re-run an experiment with exactly the same settings. This is especially important if you’re collaborating with teammates or if you revisit a project after some time.

**Separation of Concerns**:
Storing configurations outside of your training script keeps the script clean and focused on the model logic. It also lets you keep different configuration files (e.g., `parameters_dev.yml`, `parameters_prod.yml`) to easily switch between environments.

**Documentation**:
A well-structured config file serves as a form of documentation. Anyone looking at parameters.yml can quickly understand what hyperparameters, paths, or model settings your pipeline expects.

**Scalability**:
As your project grows, you might add more parameters (such as data augmentation settings, advanced model hyperparameters, etc.). Instead of cluttering your code, you simply expand your config file.