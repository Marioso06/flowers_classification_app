import argparse
import yaml
import os

# Automatically detect the root directory of the ML project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "parameters.yml")

def load_config():
    """Loads parameters from a YAML configuration file and converts relative paths to absolute."""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    # Convert relative paths to absolute paths
    for key, value in config.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):  # Check if it's a file/directory path
            if not os.path.isabs(value):  # Ensure it's relative to the project root
                config[key] = os.path.abspath(os.path.join(PROJECT_ROOT, value))

    return config

def get_input_args():
    """Parses command-line arguments with defaults loaded from a YAML file."""
    config = load_config()

    parser = argparse.ArgumentParser(description="Command-line arguments for CNN training and prediction")

    parser.add_argument('--data_directory', type=str, default=config["data_directory"], help='Path to flower images')

    parser.add_argument('--arch', type=str, default=config["arch"], choices=['vgg11', 'vgg13', 'vgg19'], help='CNN model architecture')

    parser.add_argument('--save_name', type = str, default = config["save_name"], help = 'Checkpoint file name')

    parser.add_argument('--save_dir', type=str, default=config["save_dir"], help='Folder where checkpoints are saved')

    parser.add_argument('--learning_rate', type=float, default=config["learning_rate"], help='Learning rate for training')

    parser.add_argument('--epochs', type=int, default=config["epochs"], help='Number of training epochs')

    parser.add_argument('--print_every', type=int, default=config["print_every"], help='Print training accuracy every N steps')

    parser.add_argument('--freeze_parameters', type=bool, default=config["freeze_parameters"], help='Freeze CNN parameters')

    parser.add_argument('--hidden_units', type=int, default=config["hidden_units"], help='Number of hidden units')

    parser.add_argument('--dropout', type=float, default=config["dropout"], help='Dropout probability')

    parser.add_argument('--training_compute', type=str, default=config["training_compute"], choices=['cpu', 'gpu'], help='Use CPU or GPU')

    parser.add_argument('--category_names', type=str, default=config["category_names"], help='Path to category name mapping file')

    parser.add_argument('--top_k', type=int, default=config["top_k"], help='Top K probabilities for predictions')

    parser.add_argument('--checkpoint', type=str, default=config["checkpoint"], help='Path to CNN checkpoint file')

    parser.add_argument('--image_path', type=str, default=config["image_path"], help='Path to the image for prediction')

    args = parser.parse_args()

    return args