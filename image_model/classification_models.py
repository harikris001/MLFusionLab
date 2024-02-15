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

    mnasnet1_model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=1280),
        nn.Linear(1280,512),
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

    mobilenet_small.classifier[3] = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Linear(1024,512),
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
        nn.BatchNorm1d(num_features=1920),
        nn.Linear(1920,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return densenet201_model


# Large Models 

def effnetv2small(output_classes: int):

    effnetv2_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT, progress = True)

    for param in effnetv2_model.parameters():
        param.requires_grad = True

    effnetv2_model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=1280),
        nn.Linear(1280,512),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(0.4),
        nn.Linear(512,output_classes)
    )

    return effnetv2_model


def inception(output_classes: int):

    inception_model = models.inception_v3(weights = models.Inception_V3_Weights.DEFAULT, progress = True)

    for param in inception_model.parameters():
        param.requires_grad = True

    inception_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=2048),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)
    )

    return inception_model

def resnet_50(output_classes: int):

    resenet50_model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT, progress = True)

    for param in resenet50_model.parameters():
        param.requires_grad = True

    resenet50_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=2048),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)
    )

    return resenet50_model

def regnet32gf(output_classes: int):

    regnet32gf_model = models.regnet_y_32gf(weights = models.RegNet_Y_32GF_Weights.DEFAULT, progress = True)

    for param in regnet32gf_model.parameters():
        param.requires_grad = True

    regnet32gf_model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=3172),
        nn.Linear(3172,1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)
    )

    return regnet32gf_model


def effnetb5(output_classes: int):

    effnetb5_model = models.efficientnet_b5(weights = models.EfficientNet_B5_Weights.DEFAULT, progress = True)

    for param in effnetb5_model.parameters():
        param.requires_grad = True

    effnetb5_model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=2048),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Dropout(0.4),
        nn.Linear(1024,output_classes)
    )

    return effnetb5_model