from math import fabs
import torch
import pdb
import torchvision
import torch.nn.modules
from .resnet import resnet18
import torch.nn as nn
import softpool_cuda
from SoftPool import soft_pool2d, SoftPool2d




class vgg16bn(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(vgg16bn, self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33] + model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        pretrained = False
        print("pretrained", pretrained)
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '18x':
            model = torchvision.models.resnet18(pretrained=pretrained)
            # model = resnet18()
        elif layers == '18p':
            # model = torchvision.models.resnet18(pretrained=pretrained)
            model = resnet18()
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        print(layers)
        if layers == '18x':  # pool method
            # self.maxpool = MAHPool2D(kernel_size=3, stride=2, padding=1)
            # self.maxpool=MASPool2D(kernel_size=3, stride=2, padding=1)
            # self.maxpool = SoftPool2d()
            self.maxpool = MASPool2D(kernel_size=2, stride=2)
            # self.maxpool=MAHPool2D(kernel_size=2, stride=2)
        elif layers == "18p":
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.maxpool = MASPool2D(kernel_size=2, stride=2)
        else:
            #18
            self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4
