import torch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms


imagenet_mean_pxs = np.array([0.485, 0.456, 0.406])
imagenet_std_pxs = np.array([0.229, 0.224, 0.225])

transform_params = {'resnet': {'resize_shape': 224, 'center_crop': 224},
                    'googlenet': {'resize_shape': 256, 'center_crop': 224},
                    'inceptionv3': {'resize_shape': 299, 'center_crop': 299},
                    }


def imagenet_transforms(model): 
    return transforms.Compose([
                        transforms.Resize(transform_params[model]['resize_shape']),
                        transforms.CenterCrop(transform_params[model]['center_crop']),
                        transforms.ToTensor(),
                        transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)
                    ])

def imagenet_train_transforms(model): 
    return transforms.Compose([
                transforms.RandomResizedCrop(transform_params[model]['resize_shape']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)])


jj = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)
                    ])

"""imagenet_resnet_train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)])


imagenet_googlenet_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)])

imagenet_googlenet_train_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(256),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)])"""

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    #Defines the computation performed at every call
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class GoogLeNetBottom(nn.Module):
    def __init__(self, original_model):
        super(GoogLeNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class InceptionV3Bottom(nn.Module):
    def __init__(self, original_model):
        super(InceptionV3Bottom, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:-1])#,nn.Identity())
        self.features = original_model
        self.features.fc = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])
        self.in_features = original_model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x
    
class GoogLeNetTop(nn.Module):
    def __init__(self, original_model):
        super(GoogLeNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])
        self.in_features = original_model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x
    
class InceptionV3Top(nn.Module):
    def __init__(self, original_model):
        super(InceptionV3Top, self).__init__()
        self.features = models.inception_v3(pretrained=True).fc#nn.Sequential(*[list(original_model.children())[-1]])
        self.in_features = models.inception_v3(pretrained=True).fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def set_parameter_requires_grad(model, eval_mode=True):
    if eval_mode:
        for param in model.parameters():
            param.requires_grad = False
    return

def get_model(model_name, device, get_full_model=False, eval_mode=True):
    if model_name == "resnet_18":
        model = models.resnet18(pretrained=True)
        model = model.to(device)
        train_preprocess = imagenet_train_transforms(model_name)
        val_preprocess = imagenet_transforms(model_name)
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)

    elif model_name == "resnet_50":        
        model = models.resnet50(pretrained=True)
        model = model.to(device)
        train_preprocess = imagenet_train_transforms(model_name)
        val_preprocess = imagenet_transforms(model_name)
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)
        
    elif model_name == "resnet_101":        
        model = models.resnet101(pretrained=True)
        model = model.to(device)
        train_preprocess = imagenet_train_transforms(model_name.split('_')[0])
        val_preprocess = imagenet_transforms(model_name.split('_')[0])
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)
        
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        model = model.to(device)
        train_preprocess = imagenet_train_transforms(model_name)
        val_preprocess = imagenet_transforms(model_name)
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = GoogLeNetBottom(model)
        model_top = GoogLeNetTop(model)
    
    elif model_name == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model = model.to(device)
        train_preprocess = imagenet_train_transforms(model_name)
        val_preprocess = imagenet_transforms(model_name)
        set_parameter_requires_grad(model, eval_mode)
        model_bottom = InceptionV3Bottom(model)
        model_top = InceptionV3Top(model)

    else:
        raise ValueError(model_name)

    if get_full_model:
        return model, model_bottom, model_top, train_preprocess, val_preprocess
    else:
        return model_bottom, model_top, val_preprocess

