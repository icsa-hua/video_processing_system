# Obstacle_Recognition_Edge_Ai
Obstacle Recognition 
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)

# Dummy Example: Video Processing System

## Overview
This Dummy Video Processing System is a Python-based application designed for handling video feeds from cameras, performing object detection, identifying bounding box overlaps, and communicating results via MQTT. It's a prototype suitable for understanding the flow of video processing systems.

## Components
- **Camera Module (`DummyConsumer`)**: Handles the video feed from cameras.
- **Detection Module (`DummyPredictor`)**: Simulates object detection in video frames using a dummy YOLO model.
- **Logic Module (`BoundingBoxOverlapDetector`)**: Detects overlaps among the bounding boxes of detected objects.
- **Communication Module (`DummyMQTT`)**: Sends the results to a specified MQTT topic.

## Technologies
The main technologies used for this project are: 
* Python
* Keras
* Tensorflow
* Pytorch
* Yolov5

## SetUp
1. Clone the repository:
```sh
git clone -b developer https://github.com/icsa-hua/Obstacle_Recognition_Edge_Ai.git
```
2. Navigate to the project directory:
```sh
cd Obstacle_Recognition_Edge_Ai
```
3. Install the dependencies:
```sh
pip install -r requirements.txt
```
4. To execute a program, from the project directory:
```sh
python3 test_dummy_pipeline.py
```
5. You can opt to use another model (MaskRCNN needs separate installation and is really heavy so it does not run in my machine. ONNX models produce some errors, currently working on a fix
```sh
python3 test_dummy_pipeline.py --model='Yolo' 
```
6. You can also opt to use one of two processing methods, a multithreading decomposition methods that takes every separate frame and processes it, or a streaming process which creates a batch of frames and collectively process them:
```sh
python3 test_dummy_pipeline.py  --op='streaming'
```

7. If you encounter any problem with the modules, setting the PYTHONPATH can be a solution:
```sh
export PYTHONPATH="/path to project:${PYTHONPATH}"
```

