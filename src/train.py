import torch
import os
import sys
import logging
import data_processing as dp
import mlflow
import mlflow.pytorch
import yaml
from utils.model_configuration import ModelConfiguration as mc
from utils.arg_parser import get_input_args

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self, model, criterion, optimizer, 
            device, trainloader, validloader, 
            testloader, epochs, print_every, save_directory,
            freeze_parameters,
            architecture,
            learning_rate,
            hidden_units,
            dropout,
            training_compute
            ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.epochs = epochs
        self.print_every = print_every
        self.save_directory = save_directory
        self.freeze_parameters = freeze_parameters
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.training_compute = training_compute

    def train(self):
        steps = 0
        running_loss = 0
        for epoch in range(self.epochs):
            for inputs, labels in self.trainloader:
                steps +=1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if steps % self.print_every == 0:
                    valid_loss, accuracy = self.validate()
                    mlflow.log_metric("train_loss", running_loss / self.print_every, step=steps)
                    mlflow.log_metric("valid_loss", valid_loss / len(self.validloader), step=steps)
                    mlflow.log_metric("valid_accuracy", accuracy / len(self.validloader), step=steps)

                    print(f"Epoch {epoch+1}/{self.epochs}.."
                        f"Train loss: {running_loss/self.print_every:.3f}.."
                        f"Valid loss: {valid_loss/len(self.validloader):.3f}.."
                        f"Valid accuracy: {accuracy/len(self.validloader):.3f}")
                    running_loss = 0
                    self.model.train()

    def validate(self):
        valid_loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.validloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)
                
                valid_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return valid_loss, accuracy

    def test(self):
        test_loss = 0
        accuracy = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)
                            
                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
        test_accuracy = accuracy / len(self.testloader)
        print(f"Test accuracy: {test_accuracy:.3f}")
        return test_accuracy    

    def save_checkpoint(self, epoch, class_to_idx):
        if not os.path.exists(os.path.dirname(self.save_directory)):
            os.makedirs(os.path.dirname(self.save_directory))
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_to_idx': class_to_idx,
            'freeze_parameters': self.freeze_parameters,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'hidden_units': self.hidden_units,
            'dropout': self.dropout,
            'training_compute': self.training_compute,
            "classifier": self.model.classifier
        }
        torch.save(checkpoint, self.save_directory)
        print(f"Checkpoint saved to {self.save_directory}")

if __name__ == "__main__":
    # Configure MLflow tracking URI from environment variable or use default
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
    
    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Flowers Classification App")
    
    file_path = 'data/raw.dvc'
    in_arg = get_input_args()
    print('===================== Data Preparation Started! =====================')
    #Data preprocessing
    try:
        data_preparation = dp.DataPreparation(
            data_dir=in_arg.data_directory,
            download_url='https://drive.google.com/uc?export=download&id=18I2XurHF94K072w4rM3uwVjwFpP_7Dnz'
        )
    except Exception as e:
        logger.error(f"Error initializing data preparation: {e}")
        sys.exit(1)

    try:
        data_preparation.prepare_data()
        logger.info(f"Using data directory: {in_arg.data_directory}")
        trainloader, testloader, validloader = data_preparation.transform_data()
        logger.info("Data transformation complete")
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        sys.exit(1)
    
    print('===================== Data Preparation Finished! =====================')
    
    if os.path.exists('data/raw.dvc'):
        try:
            with open('data/raw.dvc', 'r') as f:
                dvc_info = yaml.safe_load(f)
            dataset_md5 = dvc_info['outs'][0]['md5']
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            print(f"Error reading or parsing DVC file: {e}")
            dataset_md5 = "data_not_tracked"  # Or raise the exception if you prefer
        except IndexError as e:
            print(f"Error: 'outs' list is empty in the DVC file or does not contain a dictionary with 'md5': {e}")
            dataset_md5 = "data_not_tracked"
    else:
        dataset_md5 = "data_not_tracked"
    
    #Load and Get pre-trained configured model
    try:
        logger.info(f"Configuring model with architecture {in_arg.arch}, learning rate {in_arg.learning_rate}, hidden units {in_arg.hidden_units}")
        model_config = mc(in_arg.freeze_parameters, in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.dropout, in_arg.training_compute)
        model, optimizer, criterion = model_config.get_model_and_optimizer()
        logger.info(f"Model configured successfully using {model_config.device}")
    except Exception as e:
        logger.error(f"Error configuring model: {e}")
        sys.exit(1)

    #Initilize training
    save_directory = f"{in_arg.save_dir}/{in_arg.save_name}"
    logger.info(f"Model will be saved to {save_directory}")
    trainer = Trainer(model, criterion, optimizer, model_config.device, 
                      trainloader, validloader, testloader, in_arg.epochs, 
                      in_arg.print_every, save_directory,
                      in_arg.freeze_parameters, in_arg.arch, in_arg.learning_rate, 
                      in_arg.hidden_units, in_arg.dropout, in_arg.training_compute)
    # Start an MLflow run to track the experiment
    with mlflow.start_run(run_name=f"classification_{in_arg.arch}_{in_arg.training_compute}") as run:
        # Log experiment parameters from the command line or config file
        mlflow.log_params({
            "epochs": in_arg.epochs,
            "print_every": in_arg.print_every,
            "freeze_parameters": in_arg.freeze_parameters,
            "architecture": in_arg.arch,
            "learning_rate": in_arg.learning_rate,
            "hidden_units": in_arg.hidden_units,
            "dropout": in_arg.dropout,
            "training_compute": in_arg.training_compute
        })
        mlflow.set_tag("Dataset Version", dataset_md5)
        # Run training, testing, and checkpoint saving
        print('===================== Training Started! =====================')
        trainer.train()
        test_accuracy = trainer.test()
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(
            pytorch_model=trainer.model,
            artifact_path=f"model_{in_arg.arch}_{in_arg.learning_rate}_{in_arg.hidden_units}"
        )
        trainer.save_checkpoint(in_arg.epochs, class_to_idx=trainloader.dataset.class_to_idx)
        
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.end_run()
    print('===================== Training completed! =====================')
