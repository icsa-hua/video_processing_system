import numpy as np 
import cv2 
import os 
from ultralytics import YOLO 


test_path = "/mnt/c/Users/DGeorgiadis_HUA/Downloads/Fisheye8K/Fisheye8K/test/images"
output = 'Fisheye8K_Test_Normal_Yolo' 
if not os.path.exists(output): 
    os.makedirs(output) 


# yolo_model = YOLO('runs/detect/re_trained_yolov8s3/weights/best.pt')
yolo_model = YOLO('yolov8s.pt')
image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img_file in image_files:
    img_path = os.path.join(test_path, img_file)
    print(f"🔍 Processing {img_file} ...")

    # Run detection
    results = yolo_model.predict(source=img_path, conf=0.4, save=False)

    # Save annotated image
    for result in results:
        result.save(filename=os.path.join(output, img_file))

print(f"Images detections saved in {output}")
