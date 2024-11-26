import os
import cv2

def crop_images_from_labels():
    # Define the directories for test images and labels
    test_images_dir = 'high_resol_img/4096'  # Test images directory
    test_labels_dir = 'result_img/4096/label/labels'  # Directory containing label files
    img_size = 4096  # Image resolution or size info for the output directory structure

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
            label_data = label_file.readlines()

        # Load the image
        img = cv2.imread(img_path)

        # Process each label and crop the corresponding bounding box
        for i, line in enumerate(label_data):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Invalid label format in file: {label_path}")
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            x_center, y_center, width, height = (
                x_center * img.shape[1],
                y_center * img.shape[0],
                width * img.shape[1],
                height * img.shape[0]
            )

            # Calculate bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop the image to the bounding box
            cropped_img = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]

            # Create directory to save the cropped images
            save_dir = f'/data2/UROP/ljh/UROP/result/crops/{img_size}/label_{int(class_id)}'
            os.makedirs(save_dir, exist_ok=True)

            # Save the cropped image with a unique name
            save_path = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_bbox_{i}.png')
            cv2.imwrite(save_path, cropped_img)
            print(f'Saved cropped image: {save_path}')

if __name__ == '__main__':
    crop_images_from_labels()
