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
OVERLAP_AREA_THRESHOLD = 0.75  # 작은 박스가 75% 이상 겹칠 경우 필터링

# 클래스별 색상 설정
np.random.seed(42)
COLORS = {i: (np.random.randint(128, 255), np.random.randint(128, 255), np.random.randint(128, 255)) for i in range(80)}
