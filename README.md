# Obstacle_Recognition_Edge_Ai
Obstacle Recognition 
![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)


## Overview
This Video Processing System is a Python-based application designed for handling video/streaming feeds performing object detection and tracking for Road Side Perception Units (RSPUs) that remain motionless while monitoring an intersection or a highway. Detection results are transferred through MQTT to a server from a client that in later stages will be used to transmit the detection results if an abnormality is discovered. 

This uses the pretrained **You Only Look Once (YOLO)** models, for their great performance in regarsd to multiobject detection while maintaining a low GPU and memory utilization. Tracking of the objects is done with the **Bytetracker** algorithm included in the ultralytics package. 

This system saves the results, constructing the new view (with detections) in the runs/det/ directory. 


## Prerequisites

### Required software 
* Python3
* Torch (with cuda for better performance)
* Ultralytics (python package)
* paho_mqtt (python package)
* Shapely (python package)
  
Works on WSL and Linux. 
## Installation Instructions 

Clone the repository from the default branch:
```sh
git clone -b feature/streaming-ROI https://github.com/icsa-hua/Obstacle_Recognition_Edge_Ai.git
```
Navigate to the project directory:
```sh
cd Obstacle_Recognition_Edge_Ai
```
> NOTE: You should consider using a virtual environment. [Miniconda](https://docs.anaconda.com/miniconda/) is a great and easy way to handle the venv. 

Install the dependencies:
```sh
pip install -r requirements.txt
```

## Usage Instructions 
To execute a simple program execution which is recommended to test everything is functional:
```sh
python3 obs_pipeline.py --source=(path/to/video or streaming key)
```

> You can copy the following example:
```sh
python3 obs_pipeline.py --source=samples/sample_video.mp4 --show --verbose --mqtt
```

You can opt to use another model by changing the ```--name``` argument.  Models supported are YOLOv5 (all) and YOLOv8 (all). 

```sh
python3 obs_pipeline.py --source=(path/to/video) --name='yolov8s' 
```

You can opt to use a very simplistic GUI for easier visualization of inputs. 
```sh
python3 obs_pipeline.py --gui
```

You can opt to not use the MQTT broker to get better performance from the model. Just do not include the ```--mqtt``` argument on execution. 
For the same reasoning you can opt to not show the results during inference, or print out the performance from the inference of batches. Simply do not include the ```--show``` or ```--verbose``` arguments. 

Finally there is the option to use tracking which is the default program execution ```--type=tracking``` (recommended to not change). 



If you encounter any problem with the modules, setting the PYTHONPATH can be a potential solution:
```sh
export PYTHONPATH="/path to project:${PYTHONPATH}"
```

## API Documentation
The main classes of the API can be found inside the obs_system directory. Almost every sub-directory includes an interface with the generalized class and the scripts to use it. 

> Detection Module

This module is responsible for processing the video source and using batches of frames to inference them altogether, thus improving the time of execution without mitigating the accuracy. 
Includes the YOLOStreamer interface which is used to create YOLO5Streamer/YOLO8Streamer that sets up the model, with pytorch, calls the data loader based on the type of source and finally inferences batches of images. 

> Communication Module

This module creates the publisher and subscriber for an MQTT communication and transmits the performance results. This will in later stages be used to transfer batches of predictions to another server for further processing. The main class here is the RealMQTT which uses the MQTTInterface interface for 4 basic methods, connect, publish, subscribe and on_connect. 

> Application Module

This is the initial execution script to deploy the necessary resources and pipelines for the intended scenario as provided by the user. Takes the input arguments and deploys the detection model, creates the mqtt broker and configures the process. It also provides some statistics mostly for debugging and performance benchmarking. The class Application is the main object during execution that is used based on the configuration provided by the user. There is also a worker class that creates the optional simplistic GUI. 

> Logic Module

This will be used to store the ROI implementation with lane detection. At the moment this module is not utilized but can be used to detect overlaps among Bounding boxes of detected objects. 

## Examples 
## Example Result

![Streaming-ROI-cleaned](https://drive.google.com/file/d/18OKldQJ1qnvZTyHh47TX0JdCDPhtYtNr/view?usp=drive_link)

!https://drive.google.com/file/d/17MbK2pg84HQpuVNPFEPp-I8l5wc7liXi/view?usp=drive_link]


## FAQ and Troubleshooting 
1. Streaming approach is provided by the Ultralytics implementation which can be found in the documentation [here](https://docs.ultralytics.com/reference/engine/predictor/?h=stream#ultralytics.engine.predictor.BasePredictor.setup_model). This was tailored to yolov8 but we transformed it to work for yolov5 as well.
2. Why use both model architectures? 
> Having the option to interchange models and benchmark their performance is critical for applications that are aiming towards embedded AI platforms. 

3. Why have a GUI? 
> Because sometimes it is more clear for what to do?! 


