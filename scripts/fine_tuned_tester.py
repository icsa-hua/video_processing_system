import os 

from ultralytics import YOLO 


MODEL_PATH = 'compressed/yolov8_fisheye_freeze_finetuned.pt'
TEST_MODEL_PATH = 'assets/unified_dataset/images/test'

model = YOLO(MODEL_PATH)

results = model.predict(TEST_MODEL_PATH, save=True, save_txt=False, conf=0.25)

