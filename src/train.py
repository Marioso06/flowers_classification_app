import torch
import os
import data_processing as dp
from utils.model_configuration import ModelConfiguration as mc
from utils.arg_parser import get_input_args

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
                            
        print(f"Test accuracy: {accuracy/len(self.testloader):.3f}")
        return accuracy

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
            'training_compute': self.training_compute
        }
        torch.save(checkpoint, self.save_directory)
        print(f"Checkpoint saved to {self.save_directory}")

if __name__ == "__main__":
    in_arg = get_input_args()
    print('===================== Training Started! =====================')
    #Data preprocessing
    data_preparation = dp.DataPreparation(
        data_dir=in_arg.data_directory,
        download_url='https://drive.google.com/uc?export=download&id=18I2XurHF94K072w4rM3uwVjwFpP_7Dnz'
    )

    data_preparation.prepare_data()
    print(in_arg.data_directory)
    trainloader, testloader, validloader = data_preparation.transform_data()

    #Load and Get pre-trained configured model
    model_config = mc(in_arg.freeze_parameters, in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.dropout, in_arg.training_compute)
    model, optimizer, criterion = model_config.get_model_and_optimizer()

    #Initilize training
    save_directory = f"{in_arg.save_dir}/{in_arg.save_name}"
    trainer = Trainer(model, criterion, optimizer, model_config.device, 
                      trainloader, validloader, testloader, in_arg.epochs, 
                      in_arg.print_every, save_directory,
                      in_arg.freeze_parameters, in_arg.arch, in_arg.learning_rate, 
                      in_arg.hidden_units, in_arg.dropout, in_arg.training_compute)
    trainer.train()
    trainer.test()
    trainer.save_checkpoint(in_arg.epochs, class_to_idx=trainloader.dataset.class_to_idx)
    print('===================== Training completed! =====================')