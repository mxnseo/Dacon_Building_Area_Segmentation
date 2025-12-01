import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataset import SatelliteDataset
from utils.utils import rle_encode
from model.deeplabv3 import get_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=1, pretrained=False).to(device)
    
    checkpoint_path = './output/ckpt/checkpoint_30.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    test_transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    test_dataset = SatelliteDataset(
        csv_file='../data/test.csv',
        transform=test_transform, 
        infer=True
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(dummy_input)

    total_inference_time = 0
    total_images = 0
    result = []

    with torch.no_grad():
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(images)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            total_images += len(images)
            
            masks = torch.sigmoid(outputs['out']).cpu().numpy() 
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': 
                    result.append(-1)
                else:
                    result.append(mask_rle)

    avg_latency = (total_inference_time / total_images) * 1000
    fps = total_images / total_inference_time
    
    print(f"Total Images: {total_images}")
    print(f"Total Time: {total_inference_time:.4f} sec")
    print(f"Average Latency: {avg_latency:.2f} ms/image")
    print(f"FPS: {fps:.2f}")

    submit = pd.read_csv('../data/sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./deeplabv3Plus_submit.csv', index=False)
    print("Submission file created")

if __name__ == '__main__':
    main()