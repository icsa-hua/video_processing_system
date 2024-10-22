# Obstacle_Recognition_Edge_Ai
Obstacle Recognition 
![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)


## Overview
This Dummy Video Processing System is a Python-based application designed for handling video feeds from cameras, performing object detection, identifying bounding box overlaps, and communicating results via MQTT. It's a prototype suitable for understanding the flow of video processing systems. Currently utilizes batching of a video, inferencing it with yolov8 and keeping track of the objects with Bytetracker. It also creates the publisher and subscriber of the MQTT broker at the same script while transmitting correctly the performance results for each batch. This system also saves the results, constructing the new view (with detections) in the runs/det/ directory. Current GPU usage is at 1.7GB with yolov8n. Without the MQTT communication schema, performance is increased. To disable it comment the lines with self.mqtt_interface. 

## Components
- **Camera Module (`DummyConsumer`)**: Handles the source feed by passing frames through a multiprocessing pipe, with one process catching the frame and the other one processing it.  
- **Detection Module (`DummyPredictor`)**: Inference the source with Object detection model. Supports individual frame decomposition into patches for concurrent inference and reconstruction to include detections (better accuracy for overlapping) or streaming approach, where batching of a video is leveraged for higher performance. 
- **Logic Module (`BoundingBoxOverlapDetector`)**: Detects overlaps among the bounding boxes of detected objects.
- **Communication Module (`DummyMQTT`)**: Creates the publisher and subscriber for an MQTT communication and transmits the performance results. 
- **Application Module (`app`)**: Connects the distinct application components, creating the main process, mqtt broker, seting up the object detection model and using the corresponding video processing operation 

## Technologies
The main technologies used for this project are: 
* Python3.10
* Tensorflow
* Pytorch
* Yolov5/Yolov8 (Ultralytics)
* paho-mqtt
  

## SetUp
1. Clone the repository:
```sh
git clone -b feature/streaming-ROI https://github.com/icsa-hua/Obstacle_Recognition_Edge_Ai.git
```
2. Navigate to the project directory:
```sh
cd Obstacle_Recognition_Edge_Ai
```
3. Install the dependencies:
```sh
pip install -r requirements.txt
```
4. To execute the streaming choice (recommended), from the project directory:
```sh
python3 test_dummy_pipeline.py --stream --source=(path/to/video or streaming key)
```
5. You can opt to use another model. Models supported yolov5 (all), yolov8(all)(default) and yolov5(all)
```sh
python3 test_dummy_pipeline.py --name='yolov8' 
```
6. Can opt to use a very basic GUI for a more friendly experience.
```
python3 test_dummy_pipeline.py --gui
```
7. If you encounter any problem with the modules, setting the PYTHONPATH can be a potential solution:
```sh
export PYTHONPATH="/path to project:${PYTHONPATH}"
```

## Example Result
[!https://drive.google.com/file/d/17MbK2pg84HQpuVNPFEPp-I8l5wc7liXi/view?usp=drive_link](https://drive.google.com/file/d/17MbK2pg84HQpuVNPFEPp-I8l5wc7liXi/view?usp=drive_link)

