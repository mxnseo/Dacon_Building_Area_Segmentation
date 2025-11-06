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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TENSORBOARD_PATH = 'output/tensorboard'
    CHECKPOINT_DIR = 'output/ckpt'

    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    writer = SummaryWriter(TENSORBOARD_PATH)

    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    dataset = SatelliteDataset(csv_file='../data/train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = get_model(num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    global_step = 0

    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "check_point_*.pth"))
    
    if ckpt_files:
        latest_epoch = -1
        latest_ckpt_path = ""
        for ckpt_path in ckpt_files:
            match = re.search(r'check_point_(\d+).pth', ckpt_path)
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
            
            print(f"--- Resuming training from epoch {start_epoch + 1} ---")

    
    for epoch in range(start_epoch, 50):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{50}")
        
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
            pbar.set_postfix({'batch_loss': batch_loss})

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_epoch_loss}')
        
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'check_point_{epoch + 1}.pth')
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_epoch_loss
        }, checkpoint_path)
        
        print(f"--- Checkpoint saved: {checkpoint_path} ---")
            
    print("Training finished. All checkpoints saved.")
    
    writer.close()

if __name__ == '__main__':
    main()