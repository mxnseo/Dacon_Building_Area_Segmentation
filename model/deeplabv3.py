import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3, self).__init__()
        
        if pretrained:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        origin_model = deeplabv3_resnet50(weights=weights, aux_loss=True)
        
        self.backbone = origin_model.backbone
        
        self.classifier = origin_model.classifier
        self.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self.aux_classifier = origin_model.aux_classifier
        self.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        features = self.backbone(x)
        
        x = self.classifier(features['out'])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if self.training:
            aux = self.aux_classifier(features['aux'])
            aux = F.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)
            return {'out': x, 'aux': aux}
            
        return {'out': x}

def get_model(num_classes=1, pretrained=True):
    return DeepLabV3(num_classes=num_classes, pretrained=pretrained)



"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_model(num_classes=1, pretrained=True):
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    else:
        weights = None
        
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    
    model.classifier[4] = nn.Conv2d(
        256, # DeepLabv3 classifier의 마지막 in_channels
        num_classes, # (배경/건물 = 1)
        kernel_size=(1, 1),
        stride=(1, 1)
    )
    
    # 이것도 똑같이 1개 클래스로 변경
    model.aux_classifier[4] = nn.Conv2d(
        256, # DeepLabv3 aux_classifier의 마지막 in_channels
        num_classes, 
        kernel_size=(1, 1),
        stride=(1, 1)
    )
    
    return model
    
"""