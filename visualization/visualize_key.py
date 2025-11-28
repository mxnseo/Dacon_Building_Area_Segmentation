import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# RLE 디코딩 함수
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

class ImageViewer:
    def __init__(self, test_csv_path, submit_csv_path):
        self.test_df = pd.read_csv(test_csv_path)
        self.submit_df = pd.read_csv(submit_csv_path)
        self.total_imgs = len(self.test_df)
        self.current_idx = 0

        # 초기 설정
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.2) # 하단 설명 공간 확보
        
        self.update_image()
        
        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print(f"총 {self.total_imgs}개의 이미지 로드됨.")
        
        plt.show()

    def on_key(self, event):
        if event.key in ['right', 'n', ' ']:
            self.current_idx = min(self.current_idx + 1, self.total_imgs - 1)
            self.update_image()
        elif event.key in ['left', 'p']:
            self.current_idx = max(self.current_idx - 1, 0)
            self.update_image()
        elif event.key == 'q':
            plt.close()

    def update_image(self):
        # 데이터 로드
        img_path = self.test_df.iloc[self.current_idx]['img_path']
        file_name = os.path.basename(img_path)
        
        if not os.path.exists(img_path):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #
            image = cv2.resize(image, (224, 224))

        rle = self.submit_df.iloc[self.current_idx]['mask_rle']
        mask = rle_decode(rle, shape=(224, 224))
        
        # 오버레이
        overlay = image.copy()
        overlay[mask == 1] = [255, 0, 0] 
        result_overlay = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        # 화면 갱신
        for ax in self.axs:
            ax.clear()
            ax.axis('off')

        self.axs[0].imshow(image)
        self.axs[0].set_title("Original")
        
        self.axs[1].imshow(mask, cmap='gray')
        self.axs[1].set_title("Mask")
        
        self.axs[2].imshow(result_overlay)
        self.axs[2].set_title("Overlay")

        self.fig.suptitle(f"[{self.current_idx+1}/{self.total_imgs}] {file_name}", fontsize=14)
        self.fig.canvas.draw()

if __name__ == '__main__':
    TEST_CSV = './data/test.csv'
    SUBMIT_CSV = './deeplabv3Plus_submit.csv'
    
  
    ImageViewer(TEST_CSV, SUBMIT_CSV)