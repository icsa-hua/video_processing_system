from obs_system.application_module.dummy_application.dummy_app import Application

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #suppress warnings. 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import cv2
import warnings
import sys
import argparse
import logging
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Getting Initial Configuration...")
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--name', metavar='M', default='yolov8', help='Model to use (Yolov5, Yolov8 (Default), MaskRCNN, ONNX (yolov5, yolov8))')
    argparser.add_argument('--stream', metavar='W', action=argparse.BooleanOptionalAction, help='Operation to perform (streaming, inference, compression, detection)')
    argparser.add_argument('--s', metavar='S', default='0', help='Video Source to use - Local video path (.mp4) or webcam index (0, 1, 2, etc.)')
    argparser.add_argument('--type', metavar='T', default='tracking', help='Type of model to use (yolov5, yolov8)')

    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()
    
    # source = args.s
    config = {
         'source':'samples/10_DrivingWith.mp4', 
         'model_name':args.name, 
         'stream':args.stream, 
         'model_type':args.type
    }

    model_validation ={
        'yolov5': ('autoshape', 'y5'),
        'yolov8': ('autobackbone', 'y8'),
        'yolo5': ('autoshape', 'y5'),
        'yolo8': ('autobackbone', 'y8')
    }

    model_key = config['model_name'].lower()

    if model_key in model_validation and config['model_type']!="tracking":
        config['model_type'] = model_validation[model_key][0]
  
    logger.info("Initial Configuration Complete...\nStarting Application...")
    
    #Initialize application object
    app = Application()
    parent_conn = app.setup_process(config['source'])

    if parent_conn is None:
        raise Exception("Parent process connection not initialized")
    
    app.setup_model(model_name=config['model_name'], 
                    stream=config['stream'],
                    opt=config['model_type'])

    test_video_path = app.source
    converted_video_path = os.path.join(app.parent_path, 'converted_mp4/converted_video_1.mp4')
    
    if os.path.isfile(config['source']): 
        data = cv2.VideoCapture(test_video_path)
        length_of_film = data.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if app.model is None: 
        raise Exception("Model not initialized")
    
    logger.info("Model Initialized...") 

    app.setup_logic_module()
    
    app.setup_mqtt(topic="test/topic",
                   broker_address="test.mosquitto.org",
                   port=1883)
    
    logger.info("MQTT interface connected...")

    # Simulate publishing messages in intervals
    try:
        app.run_app(parent_conn=parent_conn, 
                    output_path=converted_video_path, 
                    save=False, 
                    model=config['model_name'], 
                    length_of_film=length_of_film)
        app.close_app()
    except KeyboardInterrupt as e:
        
        print(f"Exception caught: {e}")
        print("Terminating application...")
        app.close_app()
       

if __name__ == "__main__":
    main()
    

