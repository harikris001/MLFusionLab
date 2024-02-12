from torchvision import models
from torch import nn

def densenet121_model(output_classes: int):
    

    # Fetch Model and create
    densenet_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT, progress=True,)

    # Adjusting model for custom training
    for param in densenet_model.parameters():
        param.requires_grad = False

    densenet_model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, output_classes)
    )
    return densenet_model