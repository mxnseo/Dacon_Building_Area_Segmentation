import torch
import torch.nn.functional as F
from tqdm import tqdm

def dice_score(preds, masks, smooth=1e-6):
    """
    Dice Score (Dice Coefficient) 계산 함수
    preds: 모델 예측값 (0 또는 1)
    masks: 실제 마스크 (0 또는 1)
    """
    preds = preds.contiguous().view(-1)
    masks = masks.contiguous().view(-1)
    
    intersection = (preds * masks).sum()
    total_pixels = preds.sum() + masks.sum()
    
    dice = (2. * intersection + smooth) / (total_pixels + smooth)
    return dice.item()

def evaluate(model, dataloader, device, criterion):
    """
    [수정]
    모델을 평가하고 평균 Dice Score와 평균 Loss를 반환하는 함수
    criterion: (BCEWithLogitsLoss 등) 손실 함수
    """
    model.eval() # 모델을 평가 모드로 설정
    total_dice = 0.0
    total_loss = 0.0
    
    with torch.no_grad(): # Gradient 계산 비활성화
        pbar = tqdm(dataloader, desc="Validating", leave=False, dynamic_ncols=True)
        
        for images, masks in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device) # (B, H, W)
            masks_unsqueezed = masks.unsqueeze(1) # (B, 1, H, W)

            # 추론
            outputs = model(images)
            logits = outputs['out'] # (B, 1, H, W)
            
            # Loss 계산
            loss = criterion(logits, masks_unsqueezed)
            total_loss += loss.item()

            # Dice Score 계산
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float() # (B, 1, H, W)
            
            batch_dice = dice_score(preds, masks_unsqueezed)
            total_dice += batch_dice
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_dice': f'{batch_dice:.4f}'})

    model.train() # 모델을 다시 학습 모드로 설정
    
    avg_dice = total_dice / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    
    return avg_dice, avg_loss # 2개 값 반환