import torch
from torchvision import models
from torch import nn, optim


class ModelConfiguration:
    def __init__(self, freeze_parameters, model_name, learning_rate, hidden_units, dropout, training_compute) -> None:
        self.freeze_parameters = freeze_parameters
        self.model_name = model_name
        self.learning_rate =learning_rate
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.criterion = nn.NLLLoss()
        self.device = torch.device("cuda" if training_compute.lower() == "gpu" and torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()

    def _initialize_model(self):
        model = getattr(models, self.model_name)(pretrained=True)
        
        if self.freeze_parameters:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(25088, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

        model.to(self.device)

        return model
    
    def get_optimizer(self):
        return optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
    
    def get_model_and_optimizer(self):
        optimizer = self.get_optimizer()
        return self.model, optimizer, self.criterion