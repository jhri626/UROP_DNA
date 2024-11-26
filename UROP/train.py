# train.py

import os
import torch
import torch.multiprocessing as mp
import random
import numpy as np
from ultralytics import YOLO


def set_seed(seed=42):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using GPUs

    # Ensure that deterministic algorithms are used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_model():
    set_seed(42)

    # YOLOv8 모델 불러오기 (pre-trained 모델 사용)
    model = YOLO('yolov8l.pt')  # yolov8n, yolov8s, yolov8m 중 선택 가능

    # 학습 시작
    model.train(
        data='/data2/UROP/ljh/UROP/data.yaml',  # YAML 파일 경로
        epochs=500,  # 학습할 에포크 수
        imgsz=1024,  # 이미지 크기
        batch=8,  # 배치 크기
        lr0=0.01,  # 초기 학습률
        device=3,  # GPU 디바이스 선택
        project='/data2/UROP/ljh/UROP/model',  # 모델 저장 경로
        name='experiment_l_500_ver2',  # 저장될 폴더 이름
        augment=True,
        degrees=10,
        scale=0.15,
        flipud=0.5,
        fliplr=0.5,
    )


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_model()
