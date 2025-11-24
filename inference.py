import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import SatelliteDataset
from utils.utils import rle_encode
# from model.deeplabv3 import get_model
from model.DeepLabV3Plus import get_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=1, pretrained=False).to(device)
    
    checkpoint_path = './output/ckpt/checkpoint_11.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() # 평가모드

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

    result = []
    with torch.no_grad():
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            
            masks = torch.sigmoid(outputs['out']).cpu().numpy() 
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': 
                    result.append(-1)
                else:
                    result.append(mask_rle)

    # Submission
    submit = pd.read_csv('../data/sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./deeplabv3Plus_submit.csv', index=False)
    print("Submission file created")

if __name__ == '__main__':
    main()