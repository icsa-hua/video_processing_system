from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from obs_system.application_module.dummy_application.dummy_app import Application
import argparse 
import logging
import os 
import sys
import cv2
import signal
import time
from multiprocessing import Process, Queue, Value 


logger = logging.getLogger("uvicorn.error")
logger.propagate = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI() 
frame_queue = Queue(maxsize=100)
video_processing = False 
process = None 
application_inst = None

# Define the flag as a shared variable
producer_ready = Value('b', False)  # Boolean flag

class VideoProcessingRequest(BaseModel): 
    video_path: str
    name_model: str
    show: bool
    mqtt: bool


def gracefully_close_server(signum, frame):
    logger.info("Terminating Server...") 
    sys.exit(0)

signal.signal(signal.SIGINT, gracefully_close_server)


def produce_images(args, config, queue, producer_flag):
    global application_inst
    obs_app = Application(logger)
    obs_app.setup_process(config['source'], args)
    try:
        model_key = config['model_name'].lower()
        obs_app.setup_model(model_name=model_key,
                        stream=config['stream'],
                        opt=config['model_type'])
        logger.info("Model Initialized...")

    except Exception as e:
        logger.error(e)
        exit(1) 
        
    if obs_app.model is None: 
        raise Exception("Model not initialized")
    
    obs_app.setup_logic_module()
    try:
        if obs_app.mqtt:
            obs_app.setup_mqtt(topic="test/topic",
                        broker_address="test.mosquitto.org",
                        port=1883)
            logger.info("MQTT interface connected...")
        else: 
            logger.info("MQTT interface not enabled.")
    except Exception as e:
        logger.error(f"Error setting up MQTT: {e}")
        exit(1)
    
    obs_app.statistics()
    application_inst = obs_app
    if isinstance(obs_app.source,str): 
        obs_app.streamer(source=obs_app.source,
                     model=config['model_name'],
                     stream=obs_app.stream,
                     mqtt_broker=obs_app.mqtt_interface, 
                     producer_flag=producer_flag, 
                     queue=queue)
        
        obs_app.close_app()
        logger.info("Finished video processing.")


def read_frames_from_queue(queue, ready_flag):
    """Helper function to read frames from the queue."""
    while not ready_flag.value:  # Wait for producer readiness
        time.sleep(0.03)

    while True:
        frame = queue.get()
        if frame is None:  # End of stream
            break
        _, encoded_image = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               encoded_image.tobytes() + b"\r\n")


@app.post("/")
def start_video_processing(request: VideoProcessingRequest, background_tasks:BackgroundTasks):
    global video_processing, frame_queue
    video_path = request.video_path
    model_name = request.name_model
    show = request.show
    mqtt = request.mqtt
    
    if video_processing:
        return {"status": "Already running"}

    video_processing = True

    background_tasks.add_task(dummy_processing, video_path, model_name, show, mqtt)
    return {'status':"Processing started", "model": model_name, "video_path": video_path}

    

@app.post("/shutdown")
def stop_video_processing():
    global video_processing, process, application_inst
    if not video_processing:
        return {"status": "Not running"}
    video_processing = False
    if application_inst is not None: 
        application_inst.close_app()
    if process and process.is_alive(): 
        process.terminate()
        process.join()

    logger.info("Terminating application...")

    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "Server shutting down"}


@app.get("/video_feed")
def get_frame():
    global frame_queue
    logger.info("Getting frame from queue...")
    return StreamingResponse(read_frames_from_queue(frame_queue, producer_ready),
                             media_type="multipart/x-mixed-replace; boundary=frame")


def dummy_processing(video_path, model_name, show, mqtt):
    global process
    args = argparse.Namespace(name=model_name, source=video_path, type="tracking", gui=True, mqtt=mqtt, show=show, verbose=False)
    args.name = model_name.lower().split()[0]
    config = {
        'model_name':args.name,
        'stream':True,
        'source':args.source,
        'model_type':args.type
    }
    
    model_validation = {
        'yolo': ('autoshape', 'y5'),
        'yolov5': ('autoshape', 'y5'),
        'yolov8': ('autobackbone', 'y8'),
        'yolov5s': ('autoshape', 'y5'),
        'yolov8s': ('autobackbone', 'y8'),
        'yolov5n': ('autoshape', 'y5'),
        'yolov8n': ('autobackbone', 'y8'),
        'yolo5': ('autoshape', 'y5'),
        'yolo8': ('autobackbone', 'y8'),
        'yolov5m': ('autoshape', 'y5'),
        'yolov8m': ('autobackbone', 'y8')
    } 

    model_key = config['model_name'].lower()

    if model_key in model_validation and config['model_type']!="tracking":
        config['model_type'] = model_validation[model_key][0]

    process = Process(target=produce_images, args=(args, config, frame_queue, producer_ready))
    process.start()    
    
         