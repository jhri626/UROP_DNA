# main.py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'codes'))

import argparse
from ultralytics import YOLO
import cv2
from collections import Counter
from config import (
    DEFAULT_MODEL_PATH, DEFAULT_DEVICE_NUMBER, DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD, DEFAULT_IMAGE_FOLDER_PATH, DEFAULT_OUTPUT_FOLDER_PATH,
    DEFAULT_LABEL_FOLDER_PATH, COLORS, MIN_BOX_SIZE
)
from filtering import filter_boxes
from codes.utils import draw_bounding_box
from validation import load_ground_truth, evaluate  # Validation 관련 함수
import numpy as np



# CLI 인자 설정
parser = argparse.ArgumentParser(description="YOLO Model Prediction or Validation")
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help="YOLO 모델 파일 경로")
parser.add_argument('--device', type=int, default=DEFAULT_DEVICE_NUMBER, help="사용할 GPU 번호")
parser.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence Threshold")
parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD, help="NMS IoU Threshold")
parser.add_argument('--image_folder', type=str, default=DEFAULT_IMAGE_FOLDER_PATH, help="이미지 폴더 경로")
parser.add_argument('--output_folder', type=str, default=DEFAULT_OUTPUT_FOLDER_PATH, help="결과 이미지 저장 폴더")
parser.add_argument('--label_folder', type=str, default=DEFAULT_LABEL_FOLDER_PATH, help="클래스 개수 저장 폴더")
parser.add_argument('--true_label_folder', type=str, default="test/labels", help="test시 레이블")
parser.add_argument('--mode', type=str, choices=['predict', 'test'], default='predict', help="작업 모드: 'predict' 또는 'test'")
parser.add_argument('--draw_label', action='store_true', default=False, help="바운딩 박스에 클래스 라벨을 표시")
parser.add_argument('--min_box_size', type=int , default=MIN_BOX_SIZE , help="min box size")
parser.add_argument('--img_size', type=int , default=1024 , help="img size")
args = parser.parse_args()

# 결과 폴더 생성
os.makedirs(args.output_folder, exist_ok=True)
os.makedirs(args.label_folder, exist_ok=True)

# YOLO 모델 로드
model = YOLO(args.model_path)

# Validation 결과를 저장할 변수 초기화 (테스트 모드에서만 사용)
all_metrics = []

# 이미지 파일에 대해 예측 수행
for filename in os.listdir(args.image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일만 필터링
        image_path = os.path.join(args.image_folder, filename)
        
        # 이미지 예측 수행
        results = model.predict(
            source=image_path, 
            batch=1, 
            device=f"cuda:{args.device}",
            conf=args.confidence,
            iou=args.iou
            #imgsz=2048
        )
        
        # 원본 이미지 불러오기
        image = cv2.imread(image_path)
        
        # 클래스별 개수 집계 초기화
        class_counts = Counter()

        # YOLO 형식으로 저장할 데이터를 담을 리스트
        yolo_format_data = []
            
        
        # 박스 필터링 수행
        for result in results:
            filtered_boxes = filter_boxes(result.boxes)
            #filtered_boxes=result.boxes
            
            # 예측 결과를 저장할 리스트
            predictions = []
            
            for box in filtered_boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                color = COLORS[class_id]
                
                # Bounding Box 좌표 추출 및 크기 확인
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width_box, height_box = x2 - x1, y2 - y1
                width, height = args.img_size, args.img_size
                if width_box < args.min_box_size or height_box < args.min_box_size:
                    continue
                
                # 클래스별 개수 증가
                class_counts[class_name] += 1
                
                
                # YOLO 형식 데이터 변환
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                normalized_width = width_box / width
                normalized_height = height_box / height
                predictions.append((class_id, [x_center, y_center, normalized_width, normalized_height]))
                
                yolo_format_data.append(f"{class_id} {x_center:.10f} {y_center:.10f} {normalized_width:.10f} {normalized_height:.10f}")
                
                draw_bounding_box(image, (int(x1), int(y1), int(x2), int(y2)), class_name, color, args.draw_label)
                
        yolo_label_path = os.path.join(args.label_folder,"labels", f"{filename.split('.')[0]}.txt")
        directory_path = os.path.dirname(yolo_label_path)

# Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

        with open(yolo_label_path, "w") as yolo_file:
            yolo_file.write("\n".join(yolo_format_data))
        print(f"YOLO formatted labels saved to {yolo_label_path}")

        # 'test' 모드인 경우, 예측 결과와 Ground Truth 비교
        if args.mode == 'test': # 이건 안된다...
            # Ground Truth 불러오기
            ground_truths = load_ground_truth(args.true_label_folder, filename)
            
            # 성능 평가
            metrics = evaluate(predictions, ground_truths)
            all_metrics.append(metrics)
            print(f"Metrics for {filename}: {metrics}")

        # 'predict' 또는 'test' 공통 처리 (이미지 및 클래스별 개수 저장)
        output_path = os.path.join(args.output_folder, f"predicted_{filename}")
        cv2.imwrite(output_path, image)
        print(f"Predicted image saved to {output_path}")

        label_path = os.path.join(args.label_folder,"cls",f"{filename.split('.')[0]}_counts.txt")
        directory_path = os.path.dirname(label_path)

# Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            
        with open(label_path, "w") as f:
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")
        print(f"Class counts saved to {label_path}")

# 전체 평균 성능 출력 (테스트 모드에서만 수행)
if args.mode == 'test' and all_metrics:
    average_metrics = {
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "f1_score": np.mean([m["f1_score"] for m in all_metrics])
    }
    print(f"Overall Metrics: {average_metrics}")
