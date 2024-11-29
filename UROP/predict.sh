#!/bin/bash

# �⺻ ���� (�ʿ信 ���� ���� ����)
MODEL_PATH='/data2/UROP/ljh/UROP/model/experiment_l_500/weights/best.pt'  # YOLO �� ���� ���
DEVICE=3  # GPU ��ȣ (��: 0)
CONFIDENCE=0.1  # Confidence Threshold
IOU=0.7  # IoU Threshold
IMAGE_FOLDER="./test/images"  # �̹��� ���� ���
OUTPUT_FOLDER="./result_img/model_l_500_ver2_no_fil/image"  # ���� ��� �̹��� ���� ����
LABEL_FOLDER="./result_img/model_l_500_ver2_no_fil/label"  # Ŭ���� ���� ���� ����
MODE="test"  # ��� ����: predict �Ǵ� test
DRAW_LABEL=false  # �ٿ�� �ڽ��� �� ǥ�� ����: true �Ǵ� false
MIN_BOX_SIZE=20
IMG_SIZE=1024
FILTER=false


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
  $( [ "$FILTER" = true ] && echo "--filter" ) \
  $( [ "$DRAW_LABEL" = true ] && echo "--draw_label" )  # DRAW_LABEL�� ���� --draw_label �߰�