# codes/validation.py
import os
import glob
import numpy as np
from codes.utils import get_overlap_area

def calculate_iou(box1, box2):
    """두 박스 간의 IoU를 계산합니다."""
    overlap_area = get_overlap_area(box1, box2)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - overlap_area
    return overlap_area / union_area if union_area > 0 else 0

def load_ground_truth(label_folder, filename):
    """Ground Truth 레이블 파일을 로드합니다."""
    label_path = os.path.join(label_folder, filename.replace(".jpg", ".txt"))
    ground_truth = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])
            ground_truth.append((class_id, [x1, y1, x2, y2]))
    return ground_truth

def evaluate(predictions, ground_truths, iou_threshold=0.5):
    """모델 예측과 Ground Truth를 비교하여 Precision, Recall, F1-score를 계산합니다."""
    tp = fp = fn = 0

    for pred_class, pred_box in predictions:
        matched = False
        for gt_class, gt_box in ground_truths:
            if pred_class == gt_class:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp += 1
                    matched = True
                    break
        if not matched:
            fp += 1

    # False Negatives: Ground Truth가 있는데 검출되지 않은 경우
    fn = len(ground_truths) - tp

    # Precision, Recall, F1-score 계산
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}
