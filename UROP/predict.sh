#!/bin/bash

# �⺻ ���� (�ʿ信 ���� ���� ����)
MODEL_PATH="/data2/UROP/ljh/UROP/model/experiment_l_scale/weights/best.pt"  # YOLO �� ���� ���
DEVICE=3  # GPU ��ȣ (��: 0)
CONFIDENCE=0.1  # Confidence Threshold
IOU=0.8  # IoU Threshold
IMAGE_FOLDER="./full_images"  # �̹��� ���� ���
OUTPUT_FOLDER="./result_img/model_l_scale/image"  # ���� ��� �̹��� ���� ����
LABEL_FOLDER="./result_img/model_l_scale/label"  # Ŭ���� ���� ���� ����
MODE="predict"  # ��� ����: predict �Ǵ� test
DRAW_LABEL=false  # �ٿ�� �ڽ��� �� ǥ�� ����: true �Ǵ� false

# ���� ���
python predict.py \
  --model_path $MODEL_PATH \
  --device $DEVICE \
  --confidence $CONFIDENCE \
  --iou $IOU \
  --image_folder $IMAGE_FOLDER \
  --output_folder $OUTPUT_FOLDER \
  --label_folder $LABEL_FOLDER \
  --mode $MODE \
  $( [ "$DRAW_LABEL" = true ] && echo "--draw_label" )  # DRAW_LABEL�� ���� --draw_label �߰�