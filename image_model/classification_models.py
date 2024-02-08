from torchvision import models
from torch import nn

def densenet121_model(output_classes: int):

    # Fetch Model and create
    densenet_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT, progress=False)

    # Adjusting model for custom training
    for param in densenet_model.parameters():
        param.requires_grad = False
        densenet_model.classifier[1] = nn.Sequential(
            nn.BatchNorm1d(num_features=1000),
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.4),
            nn.Linear(512,output_classes),
        )
    return densenet_model