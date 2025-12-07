import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from .attention import NonLocalBlock, CBAM


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DeepLabV3_UNet_Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3_UNet_Attention, self).__init__()

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        origin_model = deeplabv3_resnet50(weights=weights, aux_loss=True)

        # Backbone
        self.backbone = origin_model.backbone
        self.backbone.return_layers = {
            'layer4': 'out',
            'layer3': 'skip3',
            'layer2': 'skip2',
            'layer1': 'skip1',
        }

        # ASPP (256ch)
        self.aspp = origin_model.classifier[0]

        # ---- Attention modules ----
        self.cbam_skip3 = CBAM(1024)               # CBAM for skip3 (1024 channels)
        self.nonlocal_block = NonLocalBlock(256)   # NonLocal after ASPP

        # Decoder blocks
        self.dec1 = DecoderBlock(256, 1024, 512, upsample=False)
        self.dec2 = DecoderBlock(512, 512, 256, upsample=True)
        self.dec3 = DecoderBlock(256, 256, 128, upsample=True)

        # Final conv head
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        # Aux classifier
        self.aux_classifier = origin_model.aux_classifier
        self.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder outputs
        features = self.backbone(x)

        # ASPP
        x_aspp = self.aspp(features['out'])     # 256ch output

        # NonLocal after ASPP
        x_aspp = self.nonlocal_block(x_aspp)

        # CBAM on skip3
        skip3 = self.cbam_skip3(features['skip3'])

        # Decoder
        x = self.dec1(x_aspp, skip3)
        x = self.dec2(x, features['skip2'])
        x = self.dec3(x, features['skip1'])

        # Final output
        x = self.final_conv(x)
        output = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        if self.training:
            aux = self.aux_classifier(features['skip3'])
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            return {"out": output, "aux": aux}

        return {"out": output}


def get_model(num_classes=1, pretrained=True):
    return DeepLabV3_UNet_Attention(num_classes=num_classes, pretrained=pretrained)
