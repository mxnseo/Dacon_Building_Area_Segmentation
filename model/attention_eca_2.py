import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import ASPP


from .attention import NonLocalBlock, ECA, CBAM


class DeepLabV3_Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()

        # Backbone with dilation
        self.backbone = resnet50(
            weights=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True],
        )

        # Remove avgpool & fc
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])

        # Channel = 2048 at layer4 output
        self.eca = ECA(2048)


        # ASPP (입력 채널 1024로 변경)
        self.aspp = ASPP(2048, (12, 24, 36))

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]

        x = self.backbone_layers(x)
        x = self.eca(x)
        x = self.aspp(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return {"out": x}


def deeplabv3_resnet50_attention(
    *,
    num_classes: int = 1,
    pretrained_backbone: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs,
):
    return DeepLabV3_Attention(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import ASPP


from .attention import NonLocalBlock, ECA, CBAM


class DeepLabV3_Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()

        # Backbone with dilation
        self.backbone = resnet50(
            weights=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True],
        )
        
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        # ECA 삽입
        self.eca2 = ECA(512)

        # ASPP 앞 bottleneck (2048 → 1024)
        self.bottleneck_in = nn.Conv2d(2048, 1024, kernel_size=1)

        # ASPP (입력 채널 1024로 변경)
        self.aspp = ASPP(1024, (12, 24, 36))

        # ASPP 뒤 bottleneck (256 → 256)
        self.bottleneck_out = nn.Conv2d(256, 256, kernel_size=1)

        # # ASPP output channels = 256
        self.cbam = CBAM(256)

        # Non-local block for global dependency
        self.nonlocal_block = NonLocalBlock(256)

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]

        x = self.layer0(x)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.eca2(x)

        x = self.layer3(x)
        x = self.layer4(x)   # 2048ch

        # ASPP 앞 bottleneck
        x = self.bottleneck_in(x)
        x = self.aspp(x)
        # ASPP 뒤 bottleneck
        x = self.bottleneck_out(x)
        x = self.cbam(x)
        x = self.nonlocal_block(x)    # Global attention
        x = self.classifier(x)

        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return {"out": x}


def deeplabv3_resnet50_attention(
    *,
    num_classes: int = 1,
    pretrained_backbone: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs,
):
    return DeepLabV3_Attention(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )

def get_model(num_classes=1, pretrained=True):

    # pretrained 사용 여부
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None

    # Attention DeepLabV3 모델 생성
    model = deeplabv3_resnet50_attention(
        num_classes=num_classes,
        pretrained_backbone=backbone_weights,
    )

    return model