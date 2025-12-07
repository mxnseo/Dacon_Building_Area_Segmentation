# Satellite Image Building Area Segmentation
https://kind-slip-86b.notion.site/2998a2c2bfdb806a86eef035defb62a8?source=copy_link

Windows11, Anaconda, RTX 5070 GPU (or RTX 4070 SUPER)

<br />

## 가상환경 구축
### GPU 5070
```text
conda create -n segmentation python=3.10 -y

conda actiavte segmemtation
```

```text
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

```text
pip install -r requirements_5070.txt
```

<br />

### GPU 4070 SUPER
```text
conda create -n segmentation python=3.10 -y

conda actiavte segmemtation
```

```text
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```text
pip install -r requirements_4070.txt
```

<br />

## Dataset

<img width="292" height="175" alt="image" src="https://github.com/user-attachments/assets/01c7f413-4851-48ca-ac11-313ee217f1d7" />

**data 디렉터리에 .csv file 포함**

<br />

### Dataset Info
- **train_img**
  - TRAIN_0000.png ~ TRAIN_7139.png
  - 1024 x 1024

<br />

- **test_img**
  - TEST_00000.png ~ TEST_60639.png
  - 224 x 224

<br />

- **train.csv**
  - img_id : 학습 위성 이미지 샘플 ID
  - img_path : 학습 위성 이미지 경로 (상대 경로)
  - mask_rle : RLE 인코딩된 이진마스크(0 : 배경, 1 : 건물) 정보

학습 위성 이미지에는 반드시 건물이 포함.

그러나 추론 위성 이미지에는 건물이 포함되어 있지 않음.

학습 위성 이미지의 촬영 해상도는 0.5m/픽셀이며, 추론 위성 이미지의 촬영 해상도는 공개하지 않음.

<br />

- **test.csv**
  - img_id : 추론 위성 이미지 샘플 ID
  - img_path : 추론 위성 이미지 경로 (상대 경로)

<br />

- **sample_submission.csv** - 제출 양식
  - img_id : 추론 위성 이미지 샘플 ID
  - mask_rle : RLE 인코딩된 예측 이진마스크(0: 배경, 1 : 건물) 정보

단, 예측 결과에 건물이 없는 경우 반드시 -1 처리


<br />

## EDA

### 총 학습 샘플: 7140
### 총 테스트 샘플: 60640

<br />


### csv 메타데이터 확인
```text
--- train.csv info (Null 값 확인) ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7140 entries, 0 to 7139
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   img_id    7140 non-null   object
 1   img_path  7140 non-null   object
 2   mask_rle  7140 non-null   object
dtypes: object(3)
memory usage: 167.5+ KB
```

<br />

### 마스크 비율 통계 및 히스토그램 시각화
<img width="1200" height="600" alt="mask_coverage_histogram" src="https://github.com/user-attachments/assets/60dc7a31-90d1-41a5-ac8d-bacf81fdec20" />

```text
--- 마스크 비율 통계 ---
count    7140.000000
mean        0.058819
std         0.043269
min         0.010000
25%         0.024270
50%         0.049321
75%         0.082361
max         0.475410
Name: mask_coverage_ratio, dtype: float64
```

<br />

### 이미지 및 마스크 시각화

마스크가 가장 작은 샘플

<img width="2400" height="800" alt="min_coverage_sample" src="https://github.com/user-attachments/assets/e6f65dc2-4ac9-4a1b-95e3-c2a5b1883c34" />

<br />

마스크가 가장 큰 샘플
<img width="2400" height="800" alt="max_coverage_sample" src="https://github.com/user-attachments/assets/347d61c3-ba82-40ca-bd16-168933e2dbe7" />

<br />

### 학습 vs 테스트 이미지 비교

<img width="2000" height="500" alt="train_test_comparison" src="https://github.com/user-attachments/assets/40d97f86-d12f-4fa7-b80f-159d763471a1" />

<br />

train에 비해 test가 더 낮은 고도에서 촬영된 것을 확인할 수 있음

resize + scaling 고려


