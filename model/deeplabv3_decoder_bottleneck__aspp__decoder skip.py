import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from torchvision.models.segmentation.deeplabv3 import ASPP


# Bottleneck
class Bottleneck1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        # 채널을 합친 후의 크기
        concat_channels = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)

class DeepLabV3_UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3_UNet, self).__init__()
        if pretrained:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        origin_model = deeplabv3_resnet50(weights=weights, aux_loss=True)
        
        # Backbone 설정
        self.backbone = origin_model.backbone
        # 중간 레이어들을 모두 뽑아옴 (나중에 결합할 거)
        self.backbone.return_layers = {
            'layer4': 'out',    # (B, 2048, H/16, W/16) -> ASPP 입력
            'layer3': 'skip3',  # (B, 1024, H/16, W/16)
            'layer2': 'skip2',  # (B, 512,  H/8,  W/8)
            'layer1': 'skip1'   # (B, 256,  H/4,  W/4)
        }

        # ASPP 앞 bottleneck (2048 → 1024)
        self.bottleneck_in = nn.Conv2d(2048, 1024, kernel_size=1)
        # self.aspp = origin_model.classifier[0] # Output: 256ch
        self.aspp = ASPP(1024, (12, 24, 36))
        # ASPP 뒤 bottleneck (256 → 256)
        self.bottleneck_out = nn.Conv2d(256, 256, kernel_size=1)


        # Skip Bottleneck 적용
        self.skip3_bottleneck = Bottleneck1x1(1024, 256)
        self.skip2_bottleneck = Bottleneck1x1(512, 128)
        self.skip1_bottleneck = Bottleneck1x1(256, 64)


        # Decoder blocks (채널 재정렬)
        # dec1: ASPP(256) + skip3(256) → 512
        self.dec1 = DecoderBlock(256, 256, 512, upsample=False)

        # dec2: 512 + skip2(128) → 256
        self.dec2 = DecoderBlock(512, 128, 256, upsample=True)

        # dec3: 256 + skip1(64) → 128
        self.dec3 = DecoderBlock(256, 64, 128, upsample=True)

        # Final head
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.aux_classifier = origin_model.aux_classifier
        self.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Encoder
        features = self.backbone(x)
        # Encoder 출력
        out = features['out']        # (B, 2048, H/16, W/16)

        # ASPP 앞 bottleneck (2048 → 1024)
        out = self.bottleneck_in(out)

        #  ASPP (1024 → 256)
        x = self.aspp(out)

        # ASPP 뒤 bottleneck (256 → 256)
        x = self.bottleneck_out(x)


        # Skip bottlenecks
        skip3 = self.skip3_bottleneck(features['skip3'])   # 256
        skip2 = self.skip2_bottleneck(features['skip2'])   # 128
        skip1 = self.skip1_bottleneck(features['skip1'])   # 64

        # Decoder
        x = self.dec1(x, skip3)   # → 512 channels
        x = self.dec2(x, skip2)   # → 256
        x = self.dec3(x, skip1)   # → 128

    
        # Final Output (x4 Upsample) 최종 결과가 원본 (1/1) 그 부분)
        x = self.final_conv(x)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if self.training:
                        aux = self.aux_classifier(features['skip3'])
                        aux = F.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)
                        return {'out': output, 'aux': aux}
            
        return {'out': output}

def get_model(num_classes=1, pretrained=True):
    return DeepLabV3_UNet(num_classes=num_classes, pretrained=pretrained)