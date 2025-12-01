import cv2
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import os
import sys


CSV_FILE_PATH = './data/train.csv'  # CSV 파일 경로
IMG_COL_NAME = 'img_path'           


class AugmentationViewer:
    def __init__(self, csv_path, img_col):
        # 1. 데이터 로드
        if not os.path.exists(csv_path):
            print(f"오류: 파일을 찾을 수 없습니다 -> {csv_path}")
            sys.exit(1)
            
        self.df = pd.read_csv(csv_path)
        self.img_col = img_col
        self.total_images = len(self.df)
        self.current_idx = 0
        
        print(f"데이터 로드 완료: 총 {self.total_images}장")

      
        self.aug_steps = [
            ("Original", A.NoOp()), 
            ("Crop (224)", A.RandomCrop(224, 224, p=1.0)),
            ("Rotate90", A.RandomRotate90(p=1.0)), 
            ("Bright/Cont", A.RandomBrightnessContrast(p=1.0, brightness_limit=0.3, contrast_limit=0.2)),
            ("Blur", A.Blur(p=1.0, blur_limit=(5, 9))),
        ]

        # 3. Matplotlib Figure 생성
        self.num_steps = len(self.aug_steps)
        self.fig, self.axes = plt.subplots(1, self.num_steps, figsize=(15, 5))
        if self.num_steps == 1: self.axes = [self.axes] # 예외처리

        # 4. 키보드 이벤트 
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 첫 이미지 렌더링
        self.update_plot()
        plt.show()

    def get_image(self, idx):
        """인덱스에 해당하는 이미지를 로드하고 RGB로 변환"""
        row = self.df.iloc[idx]
        image_path = row[self.img_col]
        
        # (필요시 경로 보정)
        # image_path = os.path.join('./data', image_path)

        image = cv2.imread(image_path)
        if image is None:
            print(f"[{idx}] 이미지 로드 실패: {image_path}")
            return None, image_path
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path

    def update_plot(self):
        """현재 인덱스의 이미지로 플롯을 업데이트"""
        image, path = self.get_image(self.current_idx)
        
        if image is None:
            # 로드 실패 시 제목에 표시하고 중단하지 않음
            self.fig.suptitle(f"[{self.current_idx+1}/{self.total_images}] Load Error: {path}", color='red')
            self.fig.canvas.draw()
            return

        # --- 증강 누적 적용 ---
        results = []
        current_image = image.copy()
        
        try:
            for name, transform in self.aug_steps:
                augmented = transform(image=current_image)
                current_image = augmented['image']
                results.append((name, current_image))
        except Exception as e:
            print(f"증강 적용 중 에러 발생: {e}")
            return

        # --- 화면 그리기 ---
        for i, (name, img_result) in enumerate(results):
            self.axes[i].clear() # 이전 그림 지우기
            self.axes[i].imshow(img_result)
            self.axes[i].set_title(f"{i+1}. {name}", fontsize=10)
            self.axes[i].axis('off')

        # 전체 제목 설정
        self.fig.suptitle(f"Image [{self.current_idx+1}/{self.total_images}]: {os.path.basename(path)}", fontsize=14)
        
        # 캔버스 갱신 (중요)
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """키보드 입력 처리 함수"""
        if event.key in ['right', ' ', 'enter']: # 다음
            self.current_idx = (self.current_idx + 1) % self.total_images
            self.update_plot()
        
        elif event.key == 'left': # 이전
            self.current_idx = (self.current_idx - 1) % self.total_images
            self.update_plot()
        
        elif event.key in ['escape', 'q']: # 종료
            print("뷰어를 종료합니다.")
            plt.close(self.fig)

if __name__ == "__main__":
    # 클래스 실행
    viewer = AugmentationViewer(CSV_FILE_PATH, IMG_COL_NAME)