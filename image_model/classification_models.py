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

def effnetb0(output_classes: int):

    effnet_bo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT, progress=True)

    for param in effnet_bo.parameters():
        param.requires_grad = False

    effnet_bo.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=1280),    
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512, output_classes),
    )

    return effnet_bo

def shuffnetv2_x0(output_classes: int):

    shufflenetv2x1 = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT, progress=True)

    for param in shufflenetv2x1.parameters():
        param.requires_grad = False

    shufflenetv2x1.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return shufflenetv2x1

def mnasNet1(output_classes: int):

    mnasnet1_model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT, progress=True)

    for param in mnasnet1_model.parameters():
        param.requires_grad = False

    mnasnet1_model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return mnasnet1_model


def MobileNetV3_small(output_classes: int):

    mobilenet_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,progress=True)

    for param in mobilenet_small.parameters():
        param.requires_grad = False

    mobilenet_small.classifier = nn.Sequential(
        nn.Linear(576,1024),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.BatchNorm1d(num_features=512),
        nn.Linear(512,output_classes)
    )
    return mobilenet_small


# Medium size models

def googleNet(output_classes : int):

    googlenet_model = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT, progress = True)

    for param in googlenet_model.parameters():
        param.requires_grad = True

    googlenet_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return googlenet_model

def regnetY16gf(output_classes: int):

    regnet_model = models.regnet_y_16gf(weights = models.RegNet_Y_16GF_Weights.DEFAULT, progress = True)

    for param in regnet_model.parameters():
        param.requires_grad = True

    regnet_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=3024),
        nn.Linear(3024,1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)   
    )

    return regnet_model

def resnet18(output_classes: int):

    resnet18_model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT, progress = True)

    for param in resnet18_model.parameters():
        param.requires_grad = True

    resnet18_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=3024),
        nn.Linear(3024,1024),
        nn.ReLU(),
        nn.BtchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)   
    )

    return resnet18_model

def effnetb3(output_classes: int):
    
    effnetb3_model = models.efficientnet_b3(weights = models.EfficientNet_B3_Weights.DEFAULT, progress = True)

    for param in effnetb3_model.parameters():
        param.requires_grad = True

    effnetb3_model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=1536),    
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512, output_classes),
    )

    return effnetb3_model

def densenet201(output_classes: int):

    densenet201_model = models.densenet201(weights = models.DenseNet201_Weights.DEFAULT, progress = True)

    for param in densenet201_model.parameters():
        param.requires_grad = True

    densenet201_model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_features=19+20),
        nn.Linear(1920,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return densenet201_model