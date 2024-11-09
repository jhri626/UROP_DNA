#!/bin/bash

# 기본 설정 (필요에 따라 변경 가능)
MODEL_PATH="/data2/UROP/ljh/UROP/model/experiment_l_scale/weights/best.pt"  # YOLO 모델 파일 경로
DEVICE=3  # GPU 번호 (예: 0)
CONFIDENCE=0.1  # Confidence Threshold
IOU=0.8  # IoU Threshold
IMAGE_FOLDER="./full_images"  # 이미지 폴더 경로
OUTPUT_FOLDER="./result_img/model_l_scale/image"  # 예측 결과 이미지 저장 폴더
LABEL_FOLDER="./result_img/model_l_scale/label"  # 클래스 개수 저장 폴더
MODE="predict"  # 모드 설정: predict 또는 test
DRAW_LABEL=false  # 바운딩 박스에 라벨 표시 여부: true 또는 false

# 실행 명령
python predict.py \
  --model_path $MODEL_PATH \
  --device $DEVICE \
  --confidence $CONFIDENCE \
  --iou $IOU \
  --image_folder $IMAGE_FOLDER \
  --output_folder $OUTPUT_FOLDER \
  --label_folder $LABEL_FOLDER \
  --mode $MODE \
  $( [ "$DRAW_LABEL" = true ] && echo "--draw_label" )  # DRAW_LABEL에 따라 --draw_label 추가