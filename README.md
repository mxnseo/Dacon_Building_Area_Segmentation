# Satellite Image Building Area Segmentation
https://dacon.io/competitions/official/236092/overview/description

Windows11, Anaconda, RTX 5070 GPU 1개

<br />

## 가상환경 구축

```text
conda create -n segmentation python=3.10 -y

conda actiavte segmemtation
```

```text
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

```text
pip install -r requirements.txt
```

<br />

## 경로 확인

<img width="292" height="175" alt="image" src="https://github.com/user-attachments/assets/01c7f413-4851-48ca-ac11-313ee217f1d7" />


data 디렉터리에 .csv file 포함
