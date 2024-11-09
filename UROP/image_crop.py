from ultralytics import YOLO
import os
import cv2

def test_model():
    # Load the trained model
    model = YOLO('/data2/UROP/ljh/UROP/model/experiment_l_500/weights/best.pt')  # Trained weights file

    # Define the directories for test images and labels
    test_images_dir = '/data2/UROP/ljh/UROP/test/images'  # Test images directory
    test_labels_dir = '/data2/UROP/ljh/UROP/test/labels'  # Test labels directory

    # Get all image paths from the test images directory
    image_paths = [os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in image_paths:
        # Get the corresponding label file path
        label_filename = os.path.basename(img_path).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(test_labels_dir, label_filename)

        # Skip if the label file doesn't exist
        if not os.path.exists(label_path):
            print(f'Label file not found for image: {img_path}')
            continue

        # Read the ground truth labels from the label file
        with open(label_path, 'r') as label_file:
            # Assuming the label file contains lines of "<class> <x> <y> <width> <height>" in YOLO format
            true_labels = [int(line.split()[0]) for line in label_file.readlines()]

        # Run prediction on the image
        results = model.predict(source=img_path, imgsz=1024, device=3)

        img = cv2.imread(img_path)

        # Iterate through each detected box for the image
        for i, (box, conf, label) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            cropped_img = img[y1:y2, x1:x2]  # Crop the image to the bounding box

            # Determine if the prediction is correct or not
            if int(label) in true_labels:
                correctness_dir = 'correct'
            else:
                correctness_dir = 'incorrect'

            # Create directory to save the cropped images if it doesn't exist
            label_dir = f'label_{int(label)}'  # Create a directory for each unique label
            save_dir = f'/data2/UROP/ljh/UROP/result/bboxes/{label_dir}/{correctness_dir}'
            os.makedirs(save_dir, exist_ok=True)

            # Save the cropped image with a unique name
            save_path = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_bbox_{i}.png')
            cv2.imwrite(save_path, cropped_img)
            print(f'Saved cropped image: {save_path}')

if __name__ == '__main__':
    test_model()
