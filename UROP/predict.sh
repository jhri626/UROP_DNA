#!/bin/bash

# �⺻ ���� (�ʿ信 ���� ���� ����)
MODEL_PATH='/data2/UROP/ljh/UROP/model/experiment_l_500/weights/best.pt'  # YOLO �� ���� ���
DEVICE=3  # GPU ��ȣ (��: 0)
CONFIDENCE=0.1  # Confidence Threshold
IOU=0.8  # IoU Threshold
IMAGE_FOLDER="./high_resol_img/4096"  # �̹��� ���� ���
OUTPUT_FOLDER="./result_img/4096/image"  # ���� ��� �̹��� ���� ����
LABEL_FOLDER="./result_img/4096/label"  # Ŭ���� ���� ���� ����
MODE="predict"  # ��� ����: predict �Ǵ� test
DRAW_LABEL=true  # �ٿ�� �ڽ��� �� ǥ�� ����: true �Ǵ� false
MIN_BOX_SIZE=80
IMG_SIZE=4096

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
  --min_box_size $MIN_BOX_SIZE \
  --img_size $IMG_SIZE \
  $( [ "$DRAW_LABEL" = true ] && echo "--draw_label" )  # DRAW_LABEL�� ���� --draw_label �߰�