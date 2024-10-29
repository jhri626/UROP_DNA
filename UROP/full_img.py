import os
import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO

# YAML 파일에서 클래스 이름을 불러오는 함수
def load_class_names(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# 클래스 레이블을 이름으로 변환하는 함수
def get_class_name(label, class_names):
    return class_names[int(label)]

# 클래스별로 고유한 색상을 지정하는 함수
def get_class_color(cls):
    class_colors = {
        0: (255, 0, 0),    # Red for class 0
        1: (0, 255, 0),    # Green for class 1
        2: (0, 0, 255),    # Blue for class 2
        3: (255, 255, 0),  # Yellow for class 3
        4: (255, 0, 255),  # Magenta for class 4
        5: (0, 255, 255),  # Cyan for class 5
        6: (128, 128, 0),  # Olive for class 6
        7: (128, 0, 128),  # Purple for class 7
        8: (0, 128, 128),  # Teal for class 8
        9: (128, 128, 128), # Gray for class 9
        10: (255, 69, 0)   # Orange for class 10
    }
    return class_colors.get(int(cls), (255, 255, 255))  # 기본값은 흰색

# 슬라이딩 윈도우 함수
def sliding_window(image, window_size, stride, output_size, edge_thickness=80, blur_ksize=(9, 9)):
    height, width = image.shape[:2]
    window_images = []
    window_coords = []

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = image[y:y + window_size, x:x + window_size]
            resized_window = cv2.resize(window, output_size, interpolation=cv2.INTER_CUBIC)
            edge_mask = np.zeros(output_size, dtype=np.uint8)

            if x != 0:
                edge_mask[:, :edge_thickness] = 1  # Left edge
            if x != width - window_size:
                edge_mask[:, -edge_thickness:] = 1  # Right edge
            if y != 0:
                edge_mask[:edge_thickness, :] = 1  # Top edge
            if y != height - window_size:
                edge_mask[-edge_thickness:, :] = 1  # Bottom edge 

            blurred_window = cv2.GaussianBlur(resized_window, blur_ksize, 0)
            weighted_window = np.where(edge_mask[:, :, np.newaxis] == 1, blurred_window, resized_window)

            window_images.append(weighted_window)
            window_coords.append((x, y, window_size, window_size))

    return window_images, window_coords

# YOLO 모델 추론 함수
def run_inference(images, model, device):
    results = []

    for img in images:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        result = model(img_tensor)
        results.append(result)

    return results

# 검출 결과를 그리는 함수 (클래스 이름 매핑 적용)
def draw_detections(image, detections, output_scale, class_names):
    height, width = image.shape[:2]
    new_width, new_height = int(width * output_scale), int(height * output_scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1 * output_scale), int(y1 * output_scale), int(x2 * output_scale), int(y2 * output_scale)

        color = get_class_color(cls)
        class_name = get_class_name(cls, class_names)  # 클래스 레이블을 이름으로 변환

        cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
        label = f'{class_name}: {conf:.2f}'
        cv2.putText(resized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return resized_image

# NMS 적용 함수
def apply_nms(detections, score_threshold=0.5, iou_threshold=0.6, min_box_size=(10, 10)):
    final_detections = []
    bboxes = [det[:4] for det in detections]
    confidences = [float(det[4]) for det in detections]
    classes = [det[5] for det in detections]

    bboxes_nms = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in bboxes]
    indices = cv2.dnn.NMSBoxes(bboxes_nms, confidences, score_threshold, iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = bboxes_nms[i]
            conf = confidences[i]
            cls = classes[i]
            if w >= min_box_size[0] and h >= min_box_size[1]:
                final_detections.append((x1, y1, x1 + w, y1 + h, conf, cls))

    def is_box_inside(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return intersection_area / box1_area if box1_area > 0 else 0

    filtered_detections = []
    for i, box1 in enumerate(final_detections):
        contained = False
        for j, box2 in enumerate(final_detections):
            if i != j:
                if is_box_inside(box1[:4], box2[:4]):
                    contained = True
                    break
                iou_with_box2 = calculate_iou(box1[:4], box2[:4])
                if iou_with_box2 > 0.6:
                    contained = True
                    break
        if not contained:
            filtered_detections.append(box1)

    return filtered_detections

# 이미지를 처리하는 함수 (클래스 이름 매핑 적용)
def process_image(input_image, window_size, stride, output_size, model, device, output_scale, class_names):
    window_images, window_coords = sliding_window(input_image, window_size, stride, output_size)

        # 슬라이딩 윈도우로 잘린 이미지를 저장할 폴더 생성
    if not os.path.exists('/home/ssdl/data2/UROP/temp_img'):
        os.makedirs('/home/ssdl/data2/UROP/temp_img')

    detections = []
    window_count = 0  # 윈도우 번호 카운트
    for img, (x, y, w, h) in zip(window_images, window_coords):
        # 윈도우 이미지 저장 경로
        window_image_path = os.path.join('/home/ssdl/data2/UROP/temp_img', f'window_{window_count}.png')
        # 이미지 저장
        cv2.imwrite(window_image_path, img)

        results = run_inference([img], model, device)

        for result in results[0]:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = result.boxes.conf[i]
                    cls = result.boxes.cls[i]
                    adjusted_x1 = x + (x1 / output_size[0]) * w
                    adjusted_y1 = y + (y1 / output_size[1]) * h
                    adjusted_x2 = x + (x2 / output_size[0]) * w
                    adjusted_y2 = y + (y2 / output_size[1]) * h
                    detections.append((adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2, conf, cls))
    window_count+=1

    final_detections = apply_nms(detections)
    output_image = draw_detections(input_image.copy(), final_detections, output_scale, class_names)

    return output_image

# 폴더 내 이미지들을 처리하는 함수
def process_images_in_folder(folder_path, window_size, stride, output_size, model, device, output_scale, output_folder, class_names):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        input_image = cv2.imread(image_path)
        input_image = cv2.resize(input_image, (512, 512))

        if input_image is None:
            print(f"Failed to load image: {image_path}")
            continue

        output_image = process_image(input_image, window_size, stride, output_size, model, device, output_scale, class_names)
        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        cv2.imwrite(output_image_path, output_image)
        print(f"Processed image saved at: {output_image_path}")

# Main script
if __name__ == "__main__":
    image_folder = '/home/ssdl/data2/UROP/full_images/'
    output_folder = '/home/ssdl/data2/UROP/processed_images/'
    yaml_file = '/home/ssdl/data2/UROP/data.yaml'  # YAML 파일 경로

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/ssdl/data2/UROP/runs/detect/train4/weights/best.pt'
    model = YOLO(model_path)
    model.to(device)

    # YAML 파일에서 클래스 이름 불러오기
    class_names = load_class_names(yaml_file)

    window_size = 128
    stride = 64
    output_size = (640, 640)
    output_scale = 4.0

    process_images_in_folder(image_folder, window_size, stride, output_size, model, device, output_scale, output_folder, class_names)
