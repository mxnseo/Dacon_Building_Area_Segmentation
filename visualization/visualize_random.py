import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

#RLE 디코딩 함수
def rle_decode(mask_rle, shape=(224, 224)):
    if mask_rle == -1 or mask_rle == '-1' or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1
    ends = starts + lengths
    
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(shape)

# 시각화 함수
def visualize_results(test_csv_path, submit_csv_path, num_samples=5):
    # 데이터 로드
    test_df = pd.read_csv(test_csv_path)
    submit_df = pd.read_csv(submit_csv_path)
    
    # 샘플 수 조정
    if num_samples > len(test_df):
        num_samples = len(test_df)

    # 랜덤 샘플 추출
    indices = random.sample(range(len(test_df)), num_samples)
    
    # plot 생성
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # 샘플이 1개일 경우 axes 차원 보정
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # 오른쪽 여백 확보 (텍스트가 잘리지 않게 하기 위함)
    plt.subplots_adjust(right=0.85)

    for i, idx in enumerate(indices):
        img_path = test_df.iloc[idx]['img_path']
        file_name = os.path.basename(img_path) # 경로 제외 파일명만 추출
        
        # 이미지 로드
        if not os.path.exists(img_path):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

        # 마스크 디코딩
        rle = submit_df.iloc[idx]['mask_rle']
        mask = rle_decode(rle, shape=(224, 224))
        
        # 오버레이
        overlay = image.copy()
        overlay[mask == 1] = [255, 0, 0]
        output = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        # --- 이미지 출력 ---
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 2].imshow(output)
        
        
        for j in range(3):
            axes[i, j].axis('off')

        
        # 맨 윗줄에만 '컬럼 제목' 표시 (Original, Mask, Overlay)
        if i == 0:
            axes[i, 0].set_title("Original Image", fontsize=14, pad=10)
            axes[i, 1].set_title("Predicted Mask", fontsize=14, pad=10)
            axes[i, 2].set_title("Overlay Result", fontsize=14, pad=10)

        # 각 행의 오른쪽 끝에 샘플 정보 표시
        info_text = f"Index: {idx}\nFile: {file_name}"
        axes[i, 2].text(1.1, 0.5, info_text, 
                        transform=axes[i, 2].transAxes, 
                        verticalalignment='center',
                        fontsize=12, fontweight='bold')

    plt.show()

#실행 
if __name__ == '__main__':
    TEST_CSV = './data/test.csv'
    SUBMIT_CSV = './deeplabv3Plus_submit.csv'
    
    # 샘플 수 지정
    num_samples = 4
    
    visualize_results(TEST_CSV, SUBMIT_CSV, num_samples)