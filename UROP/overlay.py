import cv2
import numpy as np
import os

def blend_images_in_batches_binary_and_color(folder_path, output_folder, batch_size=10):
    # Get all image files from the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                   if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Load all images (skip if unable to read)
    images = [cv2.imread(img_file) for img_file in image_files if cv2.imread(img_file) is not None]

    if not images:
        print("No valid images found in the folder.")
        return

    # Split the images into batches of batch_size
    num_batches = (len(images) + batch_size - 1) // batch_size  # To round up division

    for batch_idx in range(num_batches):
        # Get images for the current batch
        batch_images = images[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Get maximum width and height to create the base canvas for both color and binary images
        max_height = max(image.shape[0] for image in batch_images)
        max_width = max(image.shape[1] for image in batch_images)

        # Create lists to hold all aligned color and binary images
        aligned_color_images = []
        aligned_binary_images = []

        for image in batch_images:
            h, w = image.shape[:2]

            # Create a blank canvas with the max size, and place the current image in the center
            color_canvas = np.ones((max_height, max_width, 3), dtype=np.float32) * 255  # Start with white background
            binary_canvas = np.zeros((max_height, max_width), dtype=np.float32)  # Start with black background

            # Calculate top-left corner for centering
            y_offset = (max_height - h) // 2
            x_offset = (max_width - w) // 2

            # Place the current color image into the color canvas at the calculated offset
            color_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = image.astype(np.float32)

            # Convert to grayscale and apply threshold to create binary image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            # Place the current binary image into the binary canvas at the calculated offset
            binary_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = binary_image.astype(np.float32)

            # Append to the respective lists
            aligned_color_images.append(color_canvas)
            aligned_binary_images.append(binary_canvas)

        # Stack all color images along a new dimension, then take the mean across that dimension
        stacked_color_images = np.stack(aligned_color_images, axis=0)
        averaged_color_image = np.mean(stacked_color_images, axis=0)

        # Convert the final averaged color image to uint8 type for saving
        final_color_image = cv2.convertScaleAbs(averaged_color_image)

        # Stack all binary images along a new dimension, then take the mean across that dimension
        stacked_binary_images = np.stack(aligned_binary_images, axis=0)
        averaged_binary_image = np.mean(stacked_binary_images, axis=0)

        # Convert the final averaged binary image to uint8 type for saving
        final_binary_image = cv2.convertScaleAbs(averaged_binary_image)

        # Save the batch blended color and binary images
        color_output_path = os.path.join(output_folder, f'blended_batch_color_{batch_idx + 1}.jpg')
        binary_output_path = os.path.join(output_folder, f'blended_batch_binary_{batch_idx + 1}.jpg')

        cv2.imwrite(color_output_path, final_color_image)
        cv2.imwrite(binary_output_path, final_binary_image)

        print(f"Blended batch {batch_idx + 1} saved as color: {color_output_path}, binary: {binary_output_path}")

# Folder path containing images and output file path
folder_path = 'result/bboxes/label_2/correct'  # 폴더 경로를 여기에 입력하세요
output_folder = 'overlay_img/1024'  # 저장될 파일 경로 및 이름

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

blend_images_in_batches_binary_and_color(folder_path, output_folder, batch_size=150)
