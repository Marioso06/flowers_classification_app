import os
import subprocess
import torch
import logging
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GCS utils are available and import conditionally
try:
    from utils.gcs_utils import is_gcs_path, download_from_gcs, upload_to_gcs, create_gcs_directory
    HAS_GCS_SUPPORT = True
except ImportError:
    logger.warning("Google Cloud Storage utilities not available. GCS features will be disabled.")
    HAS_GCS_SUPPORT = False

class DataPreparation:
    def __init__(self, data_dir, download_url=None, bucket_name=None):
        self.data_dir = data_dir
        self.download_url = download_url
        self.bucket_name = bucket_name
        self.using_gcs = HAS_GCS_SUPPORT and bucket_name is not None
        
        # If using GCS, we need to ensure we have a local directory for temporary files
        if self.using_gcs:
            logger.info(f"Using Google Cloud Storage with bucket: {bucket_name}")
            self.local_data_dir = os.path.join(os.getcwd(), "data_temp")
            os.makedirs(self.local_data_dir, exist_ok=True)
        else:
            self.local_data_dir = self.data_dir

    def is_data_prepared(self):
        """Checks if the data directory exists and contains actual files (ignores empty subdirectories)."""
        # Check for GCS data
        if self.using_gcs:
            try:
                from utils.gcs_utils import list_gcs_files
                gcs_path = f"gs://{self.bucket_name}/{self.data_dir}"
                files = list_gcs_files(gcs_path)
                return len(files) > 0
            except Exception as e:
                logger.error(f"Error checking GCS data: {e}")
                return False
        
        # Check for local data
        if not os.path.exists(self.data_dir):
            return False  # Directory does not exist

        # Get all non-empty files (ignore empty directories)
        file_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

        return len(file_list) > 0

    def download_and_extract_data(self):
        # Create target directory if it doesn't exist
        if self.using_gcs:
            # For GCS, we'll work locally then upload
            target_dir = self.local_data_dir
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = self.data_dir
            os.makedirs(target_dir, exist_ok=True)
        
        # Download using appropriate method
        if self.using_gcs and self.bucket_name:
            self._download_with_gcs(target_dir)
        else:
            self._download_with_gdown(target_dir)
    
    def _download_with_gdown(self, target_dir):
        """Download and extract data using gdown."""
        commands = [
            f"cd {target_dir} && pip install gdown && gdown '{self.download_url}' -O flower_data.tar.gz",
            f"mkdir -p {target_dir}/flowers && cd {target_dir} && tar -xzf flower_data.tar.gz -C flowers"
        ]

        for command in commands:
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                logger.info(f"Command succeeded: {command}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                if result.stderr:
                    logger.info(f"Error Output: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed: {command}")
                logger.error(f"Return code: {e.returncode}")
                logger.error(f"Error message: {e.stderr}")
                if command == f"unlink {target_dir}":
                    continue
                else:
                    raise RuntimeError("Data preparation failed. Aborting.")
        
        # If using GCS, upload the data to the bucket
        if self.using_gcs:
            self._upload_to_gcs(target_dir)
    
    def _download_with_gcs(self, target_dir):
        """Check if data exists in GCS bucket; if not, download with gdown and upload to GCS."""
        try:
            # Try to download from GCS first
            from utils.gcs_utils import list_gcs_files, download_from_gcs
            
            gcs_data_path = f"gs://{self.bucket_name}/data"
            flower_files = list_gcs_files(f"{gcs_data_path}/flowers")
            
            if flower_files:
                logger.info(f"Found {len(flower_files)} flower files in GCS bucket")
                # Download a sample file to verify access
                sample_file = flower_files[0]
                local_sample = os.path.join(target_dir, os.path.basename(sample_file))
                download_from_gcs(sample_file, local_sample)
                logger.info(f"Successfully accessed GCS data")
                return
            else:
                logger.info("No flower data found in GCS bucket. Downloading with gdown...")
                self._download_with_gdown(target_dir)
        except Exception as e:
            logger.error(f"Error accessing GCS bucket: {e}")
            logger.info("Falling back to direct download with gdown...")
            self._download_with_gdown(target_dir)
    
    def _upload_to_gcs(self, local_dir):
        """Upload the local data directory to GCS bucket."""
        if not self.using_gcs:
            return
            
        try:
            import glob
            logger.info(f"Uploading data from {local_dir} to gs://{self.bucket_name}/data")
            
            # Create the data directories in GCS
            create_gcs_directory(f"gs://{self.bucket_name}/data")
            create_gcs_directory(f"gs://{self.bucket_name}/data/flowers")
            
            # Upload the flower_data.tar.gz file if it exists
            tar_file = os.path.join(local_dir, "flower_data.tar.gz")
            if os.path.exists(tar_file):
                upload_to_gcs(tar_file, f"gs://{self.bucket_name}/data/flower_data.tar.gz")
            
            # Upload all flower data directories
            flower_dirs = glob.glob(f"{local_dir}/flowers/*")
            for directory in flower_dirs:
                if os.path.isdir(directory):
                    dir_name = os.path.basename(directory)
                    logger.info(f"Creating directory gs://{self.bucket_name}/data/flowers/{dir_name}")
                    create_gcs_directory(f"gs://{self.bucket_name}/data/flowers/{dir_name}")
                    
                    # Upload each file in the directory
                    files = glob.glob(f"{directory}/**/*.*", recursive=True)
                    for file_path in files:
                        if os.path.isfile(file_path):
                            rel_path = os.path.relpath(file_path, local_dir)
                            gcs_path = f"gs://{self.bucket_name}/data/{rel_path}"
                            upload_to_gcs(file_path, gcs_path)
                            
            logger.info("Data upload to GCS completed successfully")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")

    def prepare_data(self):
        if self.is_data_prepared():
            logger.info("Data is already prepared. Skipping download and extraction.")
        else:
            logger.info("Data needs to be prepared. Downloading and extracting data.")
            self.download_and_extract_data()

    def transform_data(self):
        # If using GCS, we need to ensure local directories exist
        if self.using_gcs:
            # We'll work with the data we downloaded to local_data_dir
            train_dir = os.path.join(self.local_data_dir, 'flowers/train')
            valid_dir = os.path.join(self.local_data_dir, 'flowers/valid')
            test_dir = os.path.join(self.local_data_dir, 'flowers/test')
            
            # Make sure we have the directories locally
            if not (os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir)):
                logger.info("Some data directories not found locally. Downloading from GCS...")
                from utils.gcs_utils import download_from_gcs
                
                # Create local directories
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(valid_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)
                
                # Download data from GCS if necessary
                for dir_type in ['train', 'valid', 'test']:
                    gcs_dir = f"gs://{self.bucket_name}/data/flowers/{dir_type}"
                    local_dir = os.path.join(self.local_data_dir, f'flowers/{dir_type}')
                    
                    try:
                        from utils.gcs_utils import list_gcs_files
                        files = list_gcs_files(gcs_dir)
                        
                        if files:
                            for file_path in files:
                                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                                    local_file = os.path.join(local_dir, os.path.basename(file_path))
                                    download_from_gcs(file_path, local_file)
                    except Exception as e:
                        logger.error(f"Error downloading {dir_type} data from GCS: {e}")
        else:
            # Standard path handling
            train_dir = os.path.join(self.data_dir, 'flowers/train')
            valid_dir = os.path.join(self.data_dir, 'flowers/valid')
            test_dir = os.path.join(self.data_dir, 'flowers/test')
        
        logger.info(f"Data directories: {self.data_dir}")
        logger.info(f"Training directory: {train_dir}")
        
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(225),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Create image datasets
        try:
            train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
            test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
            valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
            
            logger.info(f"Found {len(train_data)} training images")
            logger.info(f"Found {len(test_data)} test images")
            logger.info(f"Found {len(valid_data)} validation images")
        except Exception as e:
            logger.error(f"Error creating image datasets: {e}")
            raise

        # Create data loaders
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

        logger.info("Data loaders created successfully.")
        return trainloader, testloader, validloader