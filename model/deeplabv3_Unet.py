import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

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
		
		# Encoder Head (기존처럼 ASPP 통과)
		self.aspp = origin_model.classifier[0] # Output: 256ch
		
		# U-Net Decoder 
		# Decoder 1: ASPP(256) + Skip3(1024) -> 1/16 크기 유지
		self.dec1 = DecoderBlock(in_channels=256, skip_channels=1024, out_channels=512, upsample=False)
		
		# Decoder 2: Dec1(512) + Skip2(512) -> 1/8 크기로 업샘플링 (x2)
		self.dec2 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256, upsample=True)
		
		# Decoder 3: Dec2(256) + Skip1(256) -> 1/4 크기로 업샘플링 (x2)
		self.dec3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128, upsample=True)
		
		# Final Head (최종 결과가 원본 (1/1) 그 부분)
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
		
		# ASPP 실행
		x_aspp = self.aspp(features['out']) # (256, H/16, W/16)
		
		# Decoder U-Net Upsampling
		# ASPP + Layer3 결합 (크기 유지)
		x = self.dec1(x_aspp, features['skip3']) # -> (512, H/16, W/16)
		
		# Layer2 결합 (x2 Upsample)
		x = self.dec2(x, features['skip2']) # -> (256, H/8, W/8)
		
		# Layer1 결합 (x2 Upsample)
		x = self.dec3(x, features['skip1']) # -> (128, H/4, W/4)
		
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