# config.py
import numpy as np

# 기본 설정 값 (main.py에서 CLI 인자로 대체 가능)
DEFAULT_MODEL_PATH = "/data2/UROP/ljh/UROP/model/experiment_l/weights/best.pt"
DEFAULT_DEVICE_NUMBER = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_IMAGE_FOLDER_PATH = "./full_images"
DEFAULT_OUTPUT_FOLDER_PATH = "./result_img/model_l/image"
DEFAULT_LABEL_FOLDER_PATH = "./result_img/model_l/label"

# Bounding Box 관련 설정
BBOX_SCALE_FACTOR = 1
FONT_SCALE = 0.5
BBOX_THICKNESS = 2
MIN_BOX_SIZE = 20
OVERLAP_AREA_THRESHOLD = 0.5  # 작은 박스가 70% 이상 겹칠 경우 필터링

# 클래스별 색상 설정
COLORS = {
    0: (255, 153, 153),  # 밝은 핑크
    1: (153, 255, 153),  # 밝은 연두
    2: (153, 153, 255),  # 밝은 파랑
    3: (255, 255, 153),  # 밝은 노랑
    4: (255, 153, 255),  # 밝은 자주
    5: (153, 255, 255),  # 밝은 하늘
    6: (255, 204, 153),  # 밝은 살구
    7: (204, 255, 153),  # 밝은 라임
    8: (153, 204, 255),  # 밝은 청색
    9: (255, 153, 204)   # 밝은 핑크빛 살구
}

