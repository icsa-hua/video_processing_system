from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.worker import MainWindow
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
    argparser.add_argument('--source', metavar='S', default=None, help='Source to use - Local video path (.mp4) or stream index (key needs to be provided)')
    argparser.add_argument('--type', metavar='T', default='tracking', help='Use tracking with bytetracker or simple detection (recommended to leave default value)')
    argparser.add_argument('--gui', metavar='G', action=argparse.BooleanOptionalAction, help='Use GUI to select video source and model')
    argparser.add_argument('--mqtt',metavar='M', action=argparse.BooleanOptionalAction, help='Use MQTT to send data to server')
    argparser.add_argument('--show', metavar='H', action=argparse.BooleanOptionalAction, help='Show real-time result inference')
    argparser.add_argument('--verbose', metavar='V', action=argparse.BooleanOptionalAction, help='Show results of inference in stdout')
    
    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()

    if args.gui:
        gui = MainWindow()
        config = gui.get_configuration()
        if config['stream'] is not None: 
            config['stream'] = True
        if config['source'] is None:
            raise ValueError("Invalid video source. Please enter a valid video source.")
        config['model_type'] = args.type
        
    else: 
        if args.source == None:
            logger.error("Invalid video source. Please enter a valid video source.")
            exit(1)

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
  
    logger.info("Initial Configuration Complete...\nStarting Application...")
    
    #Initialize application object
    app = Application()
    app.setup_process(config['source'], args)

    app.setup_model(model_name=model_key, 
                    stream=config['stream'],
                    opt=config['model_type'])

    test_video_path = app.source
    converted_video_path = os.path.join(app.parent_path, 'converted_mp4/converted_video_1.mp4')
    
    length_of_film = 0
    if os.path.isfile(config['source']): 
        data = cv2.VideoCapture(test_video_path)
        length_of_film = data.get(cv2.CAP_PROP_FRAME_COUNT)
        data.release()
        cv2.destroyAllWindows()

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
        app.run_app(output_path=converted_video_path, 
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
    

