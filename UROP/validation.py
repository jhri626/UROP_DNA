import os
import glob
import numpy as np
from collections import defaultdict
from codes.utils import get_overlap_area

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    overlap_area = get_overlap_area(box1, box2)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - overlap_area
    return overlap_area / union_area if union_area > 0 else 0

def load_ground_truth(label_folder, filename, image_width, image_height):
    """Load ground truth labels from a file and convert YOLO format to Pascal VOC."""
    label_path = os.path.join(label_folder, filename.replace(".jpg", ".txt"))
    ground_truth = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            # Convert YOLO format to Pascal VOC
            x1 = (x_center - box_width / 2) * image_width
            y1 = (y_center - box_height / 2) * image_height
            x2 = (x_center + box_width / 2) * image_width
            y2 = (y_center + box_height / 2) * image_height

            ground_truth.append((class_id, [x1, y1, x2, y2]))
    return ground_truth

def match_predictions(predictions, ground_truths, iou_threshold):
    """
    Match predictions to ground truths using a greedy approach and calculate TP, FP, and FN.
    """
    tp = 0
    fp = 0
    matched_gt = set()  # Track matched ground truth indices

    # Convert ground_truths and predictions into lists for matching
    gt_boxes = [(idx, gt_class, gt_box) for idx, (gt_class, gt_box) in enumerate(ground_truths)]
    pred_boxes = [(pred_class, pred_box) for pred_class, pred_box in predictions]

    for pred_class, pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        # Find the best matching ground truth for this prediction
        for idx, gt_class, gt_box in gt_boxes:
            if idx in matched_gt:  # Skip already matched ground truths
                continue

            if pred_class == gt_class:  # Match class
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        # Check if the match is valid
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)  # Mark the ground truth as matched
        else:
            fp += 1  # No valid match for this prediction

    # Calculate false negatives (ground truths that are not matched)
    fn = len(ground_truths) - len(matched_gt)

    print(f"tp : {tp}, fp : {fp}, fn : {fn}")

    return tp, fp, fn

def compute_precision_recall(tp, fp, fn):
    """Compute precision, recall, and F1-score."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

def evaluate(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate predictions against ground truths to calculate per-class metrics
    and overall mAP@50.
    """
    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)

    for pred_class, pred_box in predictions:
        pred_by_class[pred_class].append(pred_box)
    for gt_class, gt_box in ground_truths:
        gt_by_class[gt_class].append(gt_box)

    ap_per_class = []
    metrics_per_class = {}

    for class_id in set(pred_by_class.keys()).union(gt_by_class.keys()):
        # Get predictions and ground truths for this class
        preds = [(class_id, box) for box in pred_by_class.get(class_id, [])]
        gts = [(class_id, box) for box in gt_by_class.get(class_id, [])]

        # Match predictions and calculate TP, FP, FN
        tp, fp, fn = match_predictions(preds, gts, iou_threshold)
        precision, recall, f1_score = compute_precision_recall(tp, fp, fn)

        # Store metrics for this class
        metrics_per_class[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        # Compute AP for this class (mAP@50 is single IoU threshold)
        ap_per_class.append(precision * recall)

    # Compute mAP@50
    map50 = sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0

    precisions = [metrics['precision'] for metrics in metrics_per_class.values()]
    recalls = [metrics['recall'] for metrics in metrics_per_class.values()]
    f1_scores = [metrics['f1_score'] for metrics in metrics_per_class.values()]

    # 평균 계산
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)

    return {
        "precision": average_precision,
        "recall": average_recall,
        "f1_score": average_f1_score,
        "mAP50": map50 
    }
