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
    model = YOLO('yolov8x.pt')  # yolov8n, yolov8s, yolov8m 중 선택 가능

    # 학습 시작
    model.train(
        data='/data2/UROP/ljh/UROP/data.yaml',  # YAML 파일 경로
        epochs=300,  # 학습할 에포크 수
        imgsz=1024,  # 이미지 크기
        batch=8,  # 배치 크기
        lr0=0.01,  # 초기 학습률
        device=3,  # GPU 디바이스 선택
        augment=True,  # YOLOv8 내장 증강 활성화
        project='/data2/UROP/ljh/UROP/model',  # 모델 저장 경로
        name='experiment_x'  # 저장될 폴더 이름
    )


if __name__ == '__main__':
    train_model()
