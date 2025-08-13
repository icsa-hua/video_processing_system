# Obstacle_Recognition_Edge_Ai
Obstacle Recognition 
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)


## Overview
This Video Processing System is a Python-based application designed for handling video/streaming feeds performing object detection and tracking for Road Side Perception Units (RSPUs) that remain motionless while monitoring an intersection or a highway. Detection results are transferred through MQTT to a server from a client that in later stages will be used to transmit the detection results if an abnormality is discovered. 

This uses the pretrained **You Only Look Once (YOLO)** models, for their great performance in regarsd to multiobject detection while maintaining a low GPU and memory utilization. Tracking of the objects is done with the **Bytetracker** algorithm included in the ultralytics package. 

This system saves the results, constructing the new view (with detections) in the runs/det/ directory. 


## Prerequisites

### Utilized Software / Requirements. 
* Python --> 3.10
* Torch (with cuda for better performance)
* Ultralytics (python package)
* paho_mqtt (python package)
* Shapely (python package)
* FastAPI (python package)
* Streamlit (python package)
* ONNX - ONNXRuntime 

### Installation of Requirements 
There are two ways to go about it: 

1. Use `pip install -r requirements.txt` after cloning the module 
2. Install it as a package with: 
```sh 
pip install -e .
```

You can change what is installed with the __setup.py__. 

### Required Hardware
The program can execute with a CPU-only systems
however it is recommended that there is an 
NVIDIA GPU to significantly improve performance. 

Works on Unix-based systems.  


## Installation Instructions 

Clone the repository from the default branch:
```sh
git clone -b clean-branch https://github.com/icsa-hua/video_processing_system.git
```
Navigate to the project directory:
```sh

cd video_processing_system
```
[!] NOTE: You should consider using a virtual environment. 


## Usage Instructions 
To execute a simple program execution which is recommended to test everything is functional:
```sh
python3 obs_pipeline.py 
```
This uses the default video located in /samples

> To pass your own source of video use the following command:
```sh
python3 obs_pipeline.py --source=samples/sample_video.mp4
```

> To see the processed video in real time use the `---show` argument: 
```sh
python3 obs_pipeline.py --source=samples/sample_video.mp4 --show
```

> To see the logs and the detections returned from inference use the `---verbose` argument: 
```sh
python3 obs_pipeline.py --source=samples/sample_video.mp4 --verbose
```

> You can opt to use another model by changing the ```--name``` argument.  
[!] Models supported are YOLOv5 (all) and YOLOv8 (all) and their compressed form through ONNX. 

```sh
python3 obs_pipeline.py --source=samples/sample_video.mp4 --name='yolov8s' 
```

You can opt to use a web interface created with Streamlit and backend with Fast API. 
```sh
python3 obs_pipeline.py --gui
```

This will open a web interface to receive your inputs and 
view the processed video. An example can be seen below: 

![Screenshot 2025-01-23 104120](https://github.com/user-attachments/assets/ade0d614-d96b-4a41-b2cd-480b5755ae2f)


The pipeline initiates both the fast api server and the streamlit interface. To use the program for a stream it is recommended to use the .m3u8 stream format. 

Now if the interface was connected to the backend server, you should be able to see the results of the inference. 
![Screenshot 2025-01-23 110806](https://github.com/user-attachments/assets/03b99a2c-e4f2-4e42-910d-e2d7c57f86af)

![Screenshot 2025-01-23 110818](https://github.com/user-attachments/assets/a9e19e4a-6423-4ecd-a630-a7f8c301ed7a)

You can opt to not use the MQTT broker to get better performance from the model. Just do not include the ```--mqtt``` argument on execution. 
Using the MQTT will transmit to your designated broker information for speed (preprocessing, inference, postprocessing). 

For the same reasoning you can opt to not show the results during inference, or print out the performance from the inference of batches.
Simply do not include the ```--show``` or ```--verbose``` arguments. 

If you encounter any problem with the modules, setting the PYTHONPATH can be a potential solution:
```sh
export PYTHONPATH="/path to project:${PYTHONPATH}"
```


## Documentation
The main classes of the API can be found inside the obs_system directory. Almost every sub-directory includes an interface with the generalized class and the scripts to use it. 

> Detection Module

This module is responsible for processing the video source and using batches of frames to inference them altogether, thus improving the time of execution without mitigating the accuracy. 
Includes the YOLOStreamer interface which is used to create YOLO5Streamer/YOLO8Streamer that sets up the model, with pytorch, calls the data loader based on the type of source and finally inferences batches of images. 

> Communication Module

This module creates the publisher and subscriber for an MQTT communication and transmits the performance results.
The main class here is the RealMQTT which uses the MQTTInterface interface for 4 basic methods, connect, publish, subscribe and on_connect. 

> Application Module

This is the initial execution script to deploy the necessary resources and pipelines for the intended scenario as provided by the user.
Takes the input arguments and deploys the detection model, creates the mqtt broker and configures the process.
It also provides some statistics mostly for debugging and performance benchmarking.
The class Application is the main object during execution that is used based on the configuration provided by the user

> Logic Module

This is used to store the ROI implementation with lane detection.
We are also currently examining depth imaging algorithms but will include it on later stages. 

## FAQ and Troubleshooting 
1. Streaming approach is provided by the Ultralytics implementation which can be found in the documentation [here](https://docs.ultralytics.com/reference/engine/predictor/?h=stream#ultralytics.engine.predictor.BasePredictor.setup_model). This was tailored to yolov8 but we transformed it to work for yolov5 as well.

2. Why use both model architectures? 
> Having the option to interchange models and benchmark their performance is critical for applications that are aiming towards embedded AI platforms. 

3. Execution failed with loaders.py not containing the required methods. 
> We have modified the loaders.py file to include cropping & zooming in so that unnecessary data is not included 
making the perofmance of the model better. We include the modified file in the repo. To use it, you must copy the file 
to the ultralytics directory in your environment. Specifically, you need to copy the file to the ultralytics/data/ directory. 
Overwrite the previouus file with this new one. 

4. We can view the inference performance of the model based on profiling using pytorch Profiler. 
> We have included the profiling results in the repo in the trace_ json file. To view the results we utilize the 
viztracer package. It is included in the requirements.txt file. To use it: 
```sh
vizviewer trace_yolov8.json
```
Then open it in the browser.

5. Connection Error 
> In some cases the local host address or the port might already be in use. Make sure that no process is occupying the port. 
