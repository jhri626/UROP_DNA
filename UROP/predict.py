import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import torch

# 1. 로컬에 저장된 YOLO 모델 파일 경로 설정
model_path = "/data2/UROP/ljh/UROP/model/experiment_x/weights/best.pt"  # 사용자 학습 모델 파일 경로
device_number = 3  # 원하는 GPU 번호 (예: 1번 GPU)
model = YOLO(model_path)  # 로컬 모델 로드

# 2. 이미지 폴더 경로 설정
image_folder_path = "./full_images"  # 이미지 폴더 경로
output_folder_path = "./result_img/model_x/image"  # 결과 이미지 저장 폴더
label_folder_path = "./result_img/model_x/label"  # 클래스 개수 저장 폴더

# 결과 저장 폴더 생성
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(label_folder_path, exist_ok=True)

# Bounding Box 크기 조절 비율
bbox_scale_factor = 1  # 80%로 축소
font_scale = 0.5  # 텍스트 크기 줄이기
bbox_thickness = 2  # Bounding Box 선 두께 줄이기
min_box_size = 20  # 최소 Bounding Box 크기 (픽셀 단위, 너비 및 높이 기준)

confidence_threshold = 0.3  # Minimum confidence score
iou_threshold = 0.8  # NMS IOU threshold

# 클래스별 고유 색상 지정
num_classes = len(model.names)
np.random.seed(42)
colors = {i: (np.random.randint(128, 255), np.random.randint(128, 255), np.random.randint(128, 255)) for i in range(num_classes)}

# 3. 폴더 내 모든 이미지 파일에 대해 예측 수행
for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일만 필터링
        image_path = os.path.join(image_folder_path, filename)
        
        # 이미지 예측 수행
        results = model.predict(
            source=image_path, 
            batch=1, 
            device=f"cuda:{device_number}",
            conf=confidence_threshold,  # Confidence threshold
            iou=iou_threshold  # NMS threshold
        )
        
        # 원본 이미지 불러오기
        image = cv2.imread(image_path)
        
        # 클래스별 개수 집계 초기화
        class_counts = Counter()
        
        # 결과 이미지에 Bounding Box와 텍스트를 수동으로 그리기
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # 클래스 ID
                class_name = model.names[class_id]  # 클래스명 (모델에서 불러옴)
                color = colors[class_id]  # 클래스별 고유 색상 가져오기
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding Box 좌표

                # 축소된 Bounding Box 좌표 계산
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = (x2 - x1) * bbox_scale_factor, (y2 - y1) * bbox_scale_factor
                 
                if width < min_box_size or height < min_box_size:
                   continue  # 작은 박스는 무시
                
                x1, y1, x2, y2 = int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2)

                # 클래스별 개수 증가
                class_counts[class_name] += 1

                # Bounding Box 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=bbox_thickness)

                # 클래스 이름 텍스트 표시
                label = f"{class_name}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                label_y = max(y1, label_height + 10)
                cv2.rectangle(image, (x1, label_y - label_height - 10), (x1 + label_width, label_y + baseline - 10), color, -1)
                cv2.putText(image, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=1)

        # 각 클래스별 검출 개수 출력
        print(f"Class counts for {filename}: {class_counts}")

        # 결과 이미지 저장
        output_path = os.path.join(output_folder_path, f"predicted_{filename}")
        cv2.imwrite(output_path, image)
        print(f"Predicted image saved to {output_path}")
        
        # 클래스별 개수 txt 파일로 저장
        label_path = os.path.join(label_folder_path, f"{filename.split('.')[0]}_counts.txt")
        with open(label_path, "w") as f:
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")
        print(f"Class counts saved to {label_path}")
