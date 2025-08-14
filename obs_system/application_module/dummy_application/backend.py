from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.utils.common import *
from obs_system.utils.logger import logger 
from obs_systel.utils.appraisal import StepContext 

import os 
import sys
import cv2
import signal
import time
import argparse 
import logging

from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from multiprocessing import Process, Queue, Value 

logging.getLogger("uvicorn.error").propagate = False

process = None 
server = FastAPI() 
application_inst = None
video_processing = False 
frame_queue = Queue(maxsize=100)

# Define the flag as a shared variable
producer_ready = Value('b', False)  # Boolean flag

class VideoProcessingRequest(BaseModel): 
    video_path: str
    name_model: str
    show: bool
    mqtt: bool
    save: bool 
    verbose: bool


def gracefully_close_server(signum, frame):
    logger.info("Terminating Server...") 
    sys.exit(0)


signal.signal(signal.SIGINT, gracefully_close_server)


def produce_images(args, config, queue, producer_flag):
    global application_inst

    app = Application()

    with StepContext(name='Setup Process', catch=(KeyError, ModuleNotFoundError)):
        app.setup_process(config['source'], args)
    
    with StepContext(name='Setup Model', catch=(OSError,ValueError)):
        app.setup_model(
            model_name=config['model_name'], 
            stream=config['stream'],
            opt=config['model_type']
        )

    if app.model is None: 
        logger.error("Model not initialized")
        raise ValueError("Model not initialized")
   
    with StepContext(name='Setup Logic',catch=(KeyError,IndexError)):
        app.setup_logic_module(args) 

    with StepContext(name='Setup MQTT', catch=(ConnectionError, TimeoutError)): 
        if app.mqtt: 
            app.setup_mqtt(
                topic="test/topic", 
                broker_address="mqtt.eclipseprojects.io",
                port=1883
            )
        else: logger.debug("[MQTT] interface is disabled") 
    

    app.statistics() 
    application_inst = app 

    if isinstance(app.source,str): 
        with StepContext(name="RunApp", catch=(RuntimeError,)): 
            app.streamer(
                         source=app.source,
                         model=config['model_name'],
                         stream=app.stream,
                         mqtt_broker=app.mqtt_interface, 
                         producer_flag=producer_flag, 
                         queue=queue)
            
            app.close_app()
        
        logger.debug("-- Function produce_images finished --")


def read_frames_from_queue(queue, ready_flag):

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


@server.post("/")
def start_video_processing(request: VideoProcessingRequest, background_tasks:BackgroundTasks):
    global video_processing, frame_queue
    
    video_path = request.video_path
    model_name = request.name_model
    
    show = request.show
    mqtt = request.mqtt
    save = request.save 
    verbose = request.verbose 

    if video_processing:
        return {"status": "Already running"}

    video_processing = True

    background_tasks.add_task(dummy_processing, video_path, model_name, show, mqtt, save, verbose)
    return {'status':"Processing started", "model": model_name, "video_path": video_path}

    
@server.post("/shutdown")
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

    logger.info("-- Terminating application --")
    os.kill(os.getpid(), signal.SIGTERM)
    
    return {"status": "Server shutting down"}


@server.get("/video_feed")
def get_frame():
    global frame_queue
    logger.debug("-- Getting frame from queue --")
    return StreamingResponse(
        read_frames_from_queue(frame_queue, producer_ready),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def dummy_processing(video_path, model_name, show, mqtt, save, verbose):
    global process

    args = argparse.Namespace(name=model_name, source=video_path, type="tracking", gui=True, mqtt=mqtt, show=show, save=save, verbose=verbose)
    if " " in model_name.lower(): 
        args.name = model_name.lower().split()[0]
    else: 
        args.name = model_name.lower() 
    
    config = {
        'model_name':args.name,
        'stream':True,
        'source':args.source,
        'model_type':args.type,
        'save':args.save,
        'verbose':args.verbose       
    }

    model_key = config['model_name'].lower()
    config['model_name'] = check_model_name(model_key=model_key, condition= config['model_type'], condition_type='tracking')

    logger.info("-- Setting up process --")
    
    process = Process(target=produce_images, args=(args, config, frame_queue, producer_ready))
    process.start()    
    
         
