import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import SatelliteDataset 
from model.deeplabv3 import get_model 
from utils.validation import evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TENSORBOARD_PATH = 'output/tensorboard'
    CHECKPOINT_DIR = 'output/ckpt'
    
    
    TRAIN_CSV_PATH = '../data/train.csv'
    VAL_CSV_PATH = '../data/val.csv'
    
    EPOCHS = 100
    
    # 최신화 에퍽 코드 추가
    MAX_CKPTS_SAVED = 20 # 유지할 최대 체크포인트

    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    writer = SummaryWriter(TENSORBOARD_PATH)

    train_transform = A.Compose(
        [   
            A.RandomCrop(224, 224),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    # 일단 검증용 증강은 없는 걸로
    # 추후 TTA 고려
    val_transform = A.Compose(
        [   
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    
    train_dataset = SatelliteDataset(csv_file=TRAIN_CSV_PATH, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    val_dataset = SatelliteDataset(csv_file=VAL_CSV_PATH, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


    model = get_model(num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    global_step = 0
    best_val_dice = 0.0

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
                
            print(f"--- Restart training from epoch {start_epoch + 1} ---")

    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        
        for images, masks in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['out'], masks.unsqueeze(1)) 
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            global_step += 1
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})

        # --- 에폭 학습 종료 ---
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Average Train Loss: {avg_epoch_loss:.6f}')
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
        
        
        # 검증(Validation) 시작
        avg_val_dice = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch+1}, Average Validation Dice: {avg_val_dice:.6f}")
        writer.add_scalar('Dice/validation_epoch', avg_val_dice, epoch + 1)

        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"--- New Best Model Saved: {best_val_dice:.6f} at {best_model_path} ---")

            
        # 매 에폭 체크포인트 저장 (이어서 학습용)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_epoch_loss,
            'best_val_dice': best_val_dice # 최고 점수도 함께 저장
        }, checkpoint_path)
        
        print(f"--- Checkpoint saved: {checkpoint_path} ---") # 로그 출력
    
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