import os
import subprocess
import torch
from torchvision import datasets, transforms

class DataPreparation:
    def __init__(self, data_dir, download_url=None):
        self.data_dir = data_dir
        self.download_url = download_url

    def is_data_prepared(self):
        """Checks if the data directory exists and contains actual files (ignores empty subdirectories)."""
        if not os.path.exists(self.data_dir):
            return False  # Directory does not exist

        # Get all non-empty files (ignore empty directories)
        file_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

        return len(file_list) > 0

    def download_and_extract_data(self):
        commands = [
            f"cd {self.data_dir} && gdown '{self.download_url}' -O flower_data.tar.gz",
            f"mkdir {self.data_dir}/flowers && cd {self.data_dir} && tar -xzf flower_data.tar.gz -C flowers"
        ]

        for command in commands:
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                print(f"Command succeeded: {command}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                if result.stderr:
                    print(f"Error Output: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed: {command}")
                print(f"Return code: {e.returncode}")
                print(f"Error message: {e.stderr}")
                if command == f"unlink {self.data_dir}":
                    continue
                else:
                    raise RuntimeError("Data preparation failed. Aborting.")

    def prepare_data(self):
        if self.is_data_prepared():
            print("Data directory exists and is not empty. Skipping download and extraction.")
        else:
            print("Data directory does not exist or is empty. Downloading and extracting data.")
            self.download_and_extract_data()

    def transform_data(self):
        train_dir = os.path.join(self.data_dir, 'flowers/train')
        valid_dir = os.path.join(self.data_dir, 'flowers/valid')
        test_dir = os.path.join(self.data_dir, 'flowers/test')
        print(self.data_dir)
        print(train_dir)

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(225),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

        print("Data loaders created successfully.")
        return trainloader, testloader, validloader