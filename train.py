import os
import glob
import re
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import SatelliteDataset 
from model.deeplabv3_Unet import get_model
from utils.validation import evaluate, dice_score 


def set_seed(seed=42): # 시드 고정 함수
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42) # 시드 고정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 경로 및 설정 ---
    TENSORBOARD_PATH = 'output/tensorboard'
    CHECKPOINT_DIR = 'output/ckpt'
    
    # (학습/검증 데이터 경로 확인)
    TRAIN_CSV_PATH = 'C:/data/DBAS/data/train.csv' 
    VAL_CSV_PATH = 'C:/data/DBAS/data/val.csv'     
    
    EPOCHS = 30
    BATCH_SIZE = 16
    MAX_CKPTS_SAVED = 10 # (유지할 최대 체크포인트)
    # --- ---

    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    writer = SummaryWriter(TENSORBOARD_PATH)

    # 학습용 Transform (증강 적용)
    train_transform = A.Compose(
        [   
            A.RandomCrop(224, 224),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            A.Blur(p=1.0),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    # 검증용 Transform (증강 없음)
    val_transform = A.Compose(
        [   
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    # 학습/검증 데이터셋 및 데이터로더 분리
    train_dataset = SatelliteDataset(csv_file=TRAIN_CSV_PATH, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_dataset = SatelliteDataset(csv_file=VAL_CSV_PATH, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    model = get_model(num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)

    start_epoch = 0
    global_step = 0
    best_val_dice = 0.0

    # 체크포인트 로드 로직
    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.pth"))
    
    if ckpt_files:
        latest_epoch = -1
        latest_ckpt_path = ""
        for ckpt_path in ckpt_files:
            match = re.search(r'checkpoint_(\d+).pth', ckpt_path)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt_path = ckpt_path
        
        if latest_ckpt_path:
            print(f"Loading checkpoint: {latest_ckpt_path}")
            checkpoint = torch.load(latest_ckpt_path)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            
            if 'best_val_dice' in checkpoint:
                best_val_dice = checkpoint['best_val_dice']
                
            print(f"--- Resuming training from epoch {start_epoch + 1} ---")

    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_train_dice = 0.0 
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        
        for images, masks in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device)
            masks_unsqueezed = masks.unsqueeze(1) # (B, 1, H, W)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs['out']
            
            loss = criterion(logits, masks_unsqueezed) 
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            batch_dice = dice_score(preds, masks_unsqueezed)
            epoch_train_dice += batch_dice

            # --- TensorBoard (배치) ---
            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            # writer.add_scalar('Dice/train_batch', batch_dice, global_step) # (너무 많으면 주석 처리)
            global_step += 1
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 'batch_dice': f'{batch_dice:.4f}'})

        # --- 에폭 학습 종료 ---
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_train_dice = epoch_train_dice / len(train_dataloader) 
        
        print(f'Epoch {epoch+1}, Avg Train Loss: {avg_epoch_loss:.6f}, Avg Train Dice: {avg_epoch_train_dice:.6f}')
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
        writer.add_scalar('Dice/train_epoch', avg_epoch_train_dice, epoch + 1)
        
        
        # --- 검증(Validation) 시작 ---
        avg_val_dice, avg_val_loss = evaluate(model, val_dataloader, device, criterion)
        
        print(f"Epoch {epoch+1}, Avg Val Loss: {avg_val_loss:.6f}, Avg Val Dice: {avg_val_dice:.6f}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Dice/validation', avg_val_dice, epoch + 1)

        
        # --- 최고 점수 모델 저장 (best_model.pth) ---
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path) 
            print(f"--- New Best Model Saved with Dice: {best_val_dice:.6f} at {best_model_path} ---")

            
        # --- 매 에폭 체크포인트 저장 (이어서 학습용) ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_epoch_loss,
            'val_loss': avg_val_loss,
            'best_val_dice': best_val_dice
        }, checkpoint_path)
        
        print(f"--- Checkpoint saved: {checkpoint_path} ---")

        # --- 체크포인트 정리(Cleanup) 로직 ---
        all_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.pth"))
        if len(all_ckpts) > MAX_CKPTS_SAVED:
            ckpt_list_with_epochs = []
            
            for ckpt_path in all_ckpts:
                match = re.search(r'checkpoint_(\d+).pth', ckpt_path)
                if match:
                    epoch_num = int(match.group(1))
                    ckpt_list_with_epochs.append((epoch_num, ckpt_path))
            
            ckpt_list_with_epochs.sort(key=lambda x : x[0])
            num_to_delete = len(ckpt_list_with_epochs) - MAX_CKPTS_SAVED
            
            for epoch_num, file_path in ckpt_list_with_epochs[:num_to_delete]:
                print(f"--- Delete old checkpoint: {file_path} ---")
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error delete checkpoint {file_path}: {e}")
            
    print("Training finished. All checkpoints saved.")
    
    writer.close()

if __name__ == '__main__':
    main()