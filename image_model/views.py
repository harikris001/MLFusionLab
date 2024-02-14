import zipfile
import random
from matplotlib import pyplot as plt
import os
import io
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .classification_models import *
from .training_backend import *


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),])

from django.shortcuts import render, HttpResponseRedirect, redirect

from .models import Project

def image(request):
    if request.method == 'POST':
        pjt_name = request.POST.get('project-name')
        desc = request.POST.get('project-description')
        dataset = request.FILES['dataset']
        pjt_type = request.POST.get('project-type')
        pjt = Project(name = pjt_name, description = desc, dataset = dataset, pjt_type=pjt_type)
        pjt.save()
        print(pjt.pk)
        print(pjt_type)
        if pjt_type == 'detection':
            return HttpResponseRedirect(f'detection/?pid={pjt.pk}')
        elif pjt_type == 'classification':
            return HttpResponseRedirect(f'classification/?pid={pjt.pk}')
        elif pjt_type == 'segmentation':
            return HttpResponseRedirect(f'segmentation/?pid={pjt.pk}')
    return render(request,'image/image.html')

def detection(request):
    return render(request,'image/detection.html')

def classify(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')

        # unzipping files
        with zipfile.ZipFile('uploads/Image_datasets/tiger_dataset.zip', 'r') as zip_ref:
            zip_ref.extractall("data")

        # Getting file locations
        val_dir = os.path.join("data",'valid')
        val_data = ImageFolder(val_dir,transform=transform)

        # showing sample data
        plt.figure(figsize=(8, 2))
        fig, ax = plt.subplots(1,6)

        # collecting random images
        samples_idx = random.sample(range(len(val_data)), k=6)
            
        #iterating and getting plot
        for i, targ_sample in enumerate(samples_idx):
            targ_image, targ_label = val_data[targ_sample][0], val_data[targ_sample][1]

            targ_image_adjust = targ_image.permute(1, 2, 0)

        
            ax[i].imshow(targ_image_adjust)
            ax[i].axis("off")
            title = f"{val_data.classes[targ_label]}"
            ax[i].set_title(title, fontsize = 7)

            fig.suptitle("Sample input data")
        
        # Converting Images to IOBytes
        img_bytes = io.StringIO()
        plt.savefig(img_bytes, format='svg')
        img_bytes.seek(0)

        # Converting to Context
        img_bytes = img_bytes.getvalue()
        context = {'pid':pid, 'images':img_bytes}
        return render(request, 'image/classification.html', context)
    return render(request,'image/classification.html')

def segmentation(request):
    return render(request,'image/segementation.html')

def training(request):
    operation = request.GET.get('type')
    if operation == 'classify':
        metrics = {}
        train_data = create_train_data()
        test_data = create_test_data()
        size = request.POST.get('size')
        classes = request.POST.get('classes')
        if size == 'small':
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['MobileNet_V3_small'] = train_models(model=MobileNetV3_small(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['MNASet_1'] = train_models(model=mnasNet1(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['ShuffleNet_v2_X1'] = train_models(model=shuffnetv2_x0(output_classes=int(classes)), train_data=train_data, test_data=test_data, epochs=10)
            
            else:
                metrics['MobileNet_V3_small'] = train_models(model=MobileNetV3_small(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['MNASet_1'] = train_models(model=mnasNet1(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['ShuffleNet_v2_X1'] = train_models(model=shuffnetv2_x0(output_classes=int(classes)), train_data=train_data, test_data=test_data, epochs=10)
                metrics['DenseNet121'] = train_models(model=densenet121_model(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['EfficientNet_B0'] = train_models(model=effnetb0(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)

        elif size == 'medium':
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['GoogleNet'] = train_models(model=googleNet(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['RegNet_Y_16GF'] = train_models(model=regnetY16gf(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['ResNet18'] = train_models(model=resnet18(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
            
            else:
                metrics['EfficientNet_B3'] = train_models(model=effnetb3(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['DenseNet201'] = train_models(model=densenet201(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
        
        else:
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['GoogleNet'] = train_models(model=googleNet(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['RegNet_Y_16GF'] = train_models(model=regnetY16gf(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['ResNet18'] = train_models(model=resnet18(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
            
            else:
                metrics['EfficientNet_B3'] = train_models(model=effnetb3(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)
                metrics['DenseNet201'] = train_models(model=densenet201(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10)


        
    return render(request,'image/results.html')