import cv2
import os

def draw_bounding_boxes(image_path, label_path, output_image_path):
    print(f"이미지 경로: {image_path}")
    print(f"레이블 경로: {label_path}")

    class_colors = {
        '0': (0, 255, 255),   # 클래스 0 - 밝은 노랑
        '1': (255, 255, 0),   # 클래스 1 - 밝은 하늘색
        '2': (255, 0, 255),   # 클래스 2 - 밝은 분홍색
        '3': (0, 255, 150),   # 클래스 3 - 밝은 민트색
        '4': (150, 150, 255), # 클래스 4 - 밝은 보라색
        '5': (255, 200, 200)  # 클래스 5 - 밝은 연분홍색
    }
    
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"이미지 파일을 불러올 수 없습니다: {image_path}")
    
    image_height, image_width = image.shape[:2]
    
    # 레이블 파일 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 각 라인에서 정보를 읽고 바운딩 박스 그리기
    for line in lines:
        parts = line.strip().split()    
        cls = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # YOLO 형식의 상대 좌표를 절대 좌표로 변환
        x_center_abs = int(x_center * image_width)
        y_center_abs = int(y_center * image_height)
        width_abs = int(width * image_width)
        height_abs = int(height * image_height)
        
        # 좌상단 좌표 (x_min, y_min) 및 우하단 좌표 (x_max, y_max) 계산
        x_min = int(x_center_abs - width_abs / 2)
        y_min = int(y_center_abs - height_abs / 2)
        x_max = int(x_center_abs + width_abs / 2)
        y_max = int(y_center_abs + height_abs / 2)
        
        color = class_colors.get(cls, (255, 255, 255))
        
        # 바운딩 박스 그리기 (BGR 색상, 두께는 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 클래스 이름을 이미지에 추가 (옵션)
        label_text = f"{cls}"
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
    
    # 결과 이미지 저장
    cv2.imwrite(output_image_path, image)
    print(f"바운딩 박스가 그려진 이미지를 저장했습니다: {output_image_path}")

def process_folder(image_folder, label_folder, output_folder):
    # 폴더 내 모든 이미지 파일에 대해 처리
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png')):  # 처리할 이미지 확장자
            image_path = os.path.join(image_folder, image_file)
            
            # 레이블 파일 경로 생성
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)
            
            # 출력 이미지 경로 생성
            output_image_path = os.path.join(output_folder, f"output_{image_file}")
            
            if os.path.exists(label_path):  # 레이블 파일이 존재하는 경우에만 처리
                draw_bounding_boxes(image_path, label_path, output_image_path)
            else:
                print(f"레이블 파일이 없습니다: {label_path}")

if __name__ == "__main__":
    # 폴더 경로 설정
    image_folder = 'test/images'         # 이미지가 있는 폴더
    label_folder = 'test/labels'         # YOLO 레이블 파일이 있는 폴더
    output_folder = 'output_images/' # 출력 이미지가 저장될 폴더

    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 폴더 처리
    process_folder(image_folder, label_folder, output_folder)
