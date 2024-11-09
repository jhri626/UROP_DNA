# filtering.py
from config import OVERLAP_AREA_THRESHOLD, MIN_BOX_SIZE
from codes.utils import get_overlap_area

def filter_boxes(boxes):
    """박스 리스트에서 큰 박스 안에 있는 작은 박스를 필터링합니다."""
    filtered_boxes = []
    boxes_xyxy = [box.xyxy[0].cpu().numpy() for box in boxes]
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes_xyxy]

    for i, box in enumerate(boxes):
        contained = False
        box_coords = boxes_xyxy[i]
        box_area = areas[i]
        
        for j, other_box in enumerate(boxes):
            if i != j:
                other_box_coords = boxes_xyxy[j]
                overlap_area = get_overlap_area(box_coords, other_box_coords)
                if overlap_area / box_area > OVERLAP_AREA_THRESHOLD:
                    contained = True
                    break
        if not contained:
            filtered_boxes.append(box)
    return filtered_boxes
