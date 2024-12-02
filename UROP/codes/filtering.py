# filtering.py
from config import OVERLAP_AREA_THRESHOLD, MIN_BOX_SIZE
from codes.utils import get_overlap_area

def filter_boxes(boxes):
    """박스 리스트에서 큰 박스 안에 있는 작은 박스를 필터링합니다."""
    filtered_boxes = []
    boxes_xyxy = [box.xyxy[0].cpu().numpy() for box in boxes]
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes_xyxy]
    to_keep = [True] * len(boxes)

    for i, box in enumerate(boxes):
        if not to_keep[i]:  # 이미 필터링된 박스는 건너뜀
            continue

        box_coords = boxes_xyxy[i]
        box_area = areas[i]

        x1, y1, x2, y2 = box_coords
        width = x2 - x1
        height = y2 - y1

        if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
            to_keep[i] = False
            continue
        if width / height > 2 or width /height < 0.5:
            if width < 1.5*MIN_BOX_SIZE or height < 1.5*MIN_BOX_SIZE:
                to_keep[i]=False
            continue
        
        for j, other_box in enumerate(boxes):
            if i != j:
                other_box_coords = boxes_xyxy[j]
                other_box_area = areas[j]

                overlap_area = get_overlap_area(box_coords, other_box_coords)

                if overlap_area / min(box_area, other_box_area) > OVERLAP_AREA_THRESHOLD:
                    # 큰 박스를 유지, 작은 박스는 필터링
                    if box_area > other_box_area:
                        to_keep[j] = False  # 다른 박스 삭제
                    else:
                        to_keep[i] = False  # 현재 박스 삭제
                        break  # 현재 박스를 삭제했으므로 더 이상 비교할 필요 없음

    for i, keep in enumerate(to_keep):
        if keep:
            filtered_boxes.append(boxes[i])
    return filtered_boxes
    