# utils.py
import cv2
from config import COLORS, FONT_SCALE, BBOX_THICKNESS

def get_overlap_area(box1, box2):
    """두 박스의 겹치는 면적을 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    overlap_width = max(0, x2 - x1)
    overlap_height = max(0, y2 - y1)
    return overlap_width * overlap_height

def draw_bounding_box(image, box, class_name, color,draw_label=True):
    """이미지에 바운딩 박스를 그리고 클래스 이름을 표시합니다."""
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=BBOX_THICKNESS)
    
    if draw_label:
       # 라벨을 표시할 텍스트 박스 생성
       label = f"{class_name}"
       (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
       label_y = max(y1, label_height + 10)
       cv2.rectangle(image, (x1, label_y - label_height - 10), (x1 + label_width, label_y + baseline - 10), color, -1)
       cv2.putText(image, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), thickness=1)
