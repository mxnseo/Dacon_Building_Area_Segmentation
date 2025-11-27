import pandas as pd
import os
import numpy as np


# 원본 'train.csv' 파일의 경로
ORIGINAL_CSV_PATH = 'C:/data/DBAS/train.csv'

# 새로 생성될 학습/검증 CSV 파일 경로
OUTPUT_TRAIN_PATH = 'C:/data/DBAS/train.csv'
OUTPUT_VAL_PATH = 'C:/data/DBAS/val.csv'

# 검증 데이터셋의 비율 (0.2 = 20%)
VAL_RATIO = 0.2

# 데이터를 섞을 때 사용할 시드 (결과를 동일하게 유지하기 위함)
RANDOM_SEED = 42

def split_data():
    print(f"원본 CSV 파일 읽는 중: {ORIGINAL_CSV_PATH}")
    
    try:
        df = pd.read_csv(ORIGINAL_CSV_PATH)
    except FileNotFoundError:
        print(f"오류: '{ORIGINAL_CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("ORIGINAL_CSV_PATH 변수 경로를 올바르게 수정해주세요.")
        return
    
    print(f"총 {len(df)}개의 데이터를 찾았습니다.")

    print(f"데이터를 {1-VAL_RATIO:.0%} (학습) / {VAL_RATIO:.0%} (검증) 비율로 분리...")
    
    shuffled_df = df.sample(frac=1, random_state=RANDOM_SEED)

    val_size = int(np.ceil(len(shuffled_df) * VAL_RATIO))
    train_size = len(shuffled_df) - val_size
    
    train_df = shuffled_df.iloc[:train_size]
    val_df = shuffled_df.iloc[train_size:]
    # --- 여기까지 ---

    print(f"학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(OUTPUT_TRAIN_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 새로운 CSV 파일로 저장
    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    print(f"학습용 CSV 저장 완료: {OUTPUT_TRAIN_PATH}")
    
    val_df.to_csv(OUTPUT_VAL_PATH, index=False)
    print(f"검증용 CSV 저장 완료: {OUTPUT_VAL_PATH}")

    print("\n--- 작업 완료 ---")


if __name__ == '__main__':
    split_data()