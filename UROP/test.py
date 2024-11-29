# test.py

from ultralytics import YOLO
import os   

def test_model():
    # Load the trained model
    model = YOLO('/data2/UROP/ljh/UROP/model/experiment_l_500_ver2/weights/best.pt')  # Trained weights file

    # Run validation/test
    results = model.val(
        data='/data2/UROP/ljh/UROP/data.yaml',  # Path to YAML file
        split='test',
        imgsz=1024,  # Image size for testing
        device=3,
        batch =4,
        plots=True, 
        save_json=True, 
        project ='/data2/UROP/ljh/UROP/result',
        name = 'result_l_500_ver2_batch'
    )

    print(results.results_dict)

if __name__ == '__main__':
    test_model()
