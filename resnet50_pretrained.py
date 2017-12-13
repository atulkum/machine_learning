import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)

    model.avgpool = nn.AvgPool2d(1)
    model.fc = nn.Linear(512 * 4 * 5 * 5, num_classes)

    return model
