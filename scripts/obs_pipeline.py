from obs_system.utils.logger import get_logger, remove_logger
from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.utils.appraisal import StepContext
from obs_system.utils.common import *
from obs_system.application_module.dummy_application.intermediary import gui_connector

import os
import pdb
import cv2
import sys
import argparse

logger = get_logger(name="obs_system."+__name__)
remove_logger("matplotlib") 
remove_logger("matplotlib.font_manager") 


def main():

    logger.debug("--- Initializing Application ---")

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--model_name', metavar='M', default='trt', help='Model to use (Yolov5, Yolov8 (Default), MaskRCNN, ONNX (yolov5, yolov8))')
    argparser.add_argument('--source', metavar='SO', default='samples/highway.mp4', help='Source to use - Local video path (.mp4) or stream index (key needs to be provided)')
    argparser.add_argument('--type', metavar='T', default='tracking', help='Use tracking with bytetracker or simple detection (recommended to leave default value)')
    argparser.add_argument('--gui', metavar='G', action=argparse.BooleanOptionalAction, help='Use GUI to select video source and model')
    argparser.add_argument('--mqtt',metavar='M', action=argparse.BooleanOptionalAction, help='Use MQTT to send data to server')
    argparser.add_argument('--show', metavar='SH', action=argparse.BooleanOptionalAction, help='Show real-time result inference')
    argparser.add_argument('--verbose', metavar='V', action=argparse.BooleanOptionalAction, help='Show results of inference in stdout')
    argparser.add_argument('--port_address', metavar='P', default=8503, help='Port for Streamlit service interface')
    argparser.add_argument('--host_address', metavar='H' , default= 'localhost', help='Host server for both streamlit and fastapi')
    argparser.add_argument('--save', metavar='SA', action=argparse.BooleanOptionalAction, help='Save inference results to file')
    argparser.add_argument('--roi', metavar='R', action=argparse.BooleanOptionalAction, help='Use Region of Interest to detect obstacles')
    argparser.add_argument('--half', metavar='HF', action=argparse.BooleanOptionalAction, help='Use Half the available resources by reducing the data size (e.g. Float32 -> Float16)')
    argparser.add_argument('--fep', metavar='F', action=argparse.BooleanOptionalAction, help='Use of FishEye Projection based on camera')
    argparser.add_argument('--bench', metavar='BM', action=argparse.BooleanOptionalAction, help='Benchmark the Performance of the model and hardware.')
    argparser.add_argument('--bench-labels', metavar='BL', default='samples/labels', help="Submit the label path for GT")

    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()

    try: 
        if args.bench and not os.path.exists(args.bench_labels): 
            raise ValueError("Submit a correct path for the GT labels, that matches the video") 
    except Exception as e: 
        raise Exception(e) 
 
    logger.warning("WARNING: If you change the input video source, adjust the background subtractor image. Otherwise, it will classify all frames without movement")

    if args.gui: 
        gui_connector(args.host_address, args.port_address)   
     
    config = {
        'model_name':args.model_name,
        'stream':True, 
        'source':args.source, 
        'model_type':None, 
        'opt':args.type,
        'save':args.save if args.save is not None else False, 
        'verbose':args.verbose if args.verbose is not None else False
    }   

    model_key = config['model_name'].lower()

    # Check that the model name responds to the models approved for this application (Yolov5-v8) 
    config['model_name'], config['model_type'] = check_model_name(model_key=model_key, condition= config['opt'])

    #Initialize the application module that interfaces source, model, mqtt and logic module 
    app = Application(
        source = config['source'],
        model_name = config['model_name' ], 
        opt=config['opt'],
        model_type=config['model_type'], 
        save=config['save'], 
        verbose=config['verbose']
    )

    with StepContext(name='Setup Process', catch=(KeyError, ModuleNotFoundError)):
        app.setup_process(args) 

    with StepContext(name='Setup Model', catch=(OSError,ValueError)):
        app.setup_model(
            stream=config['stream'],
            opt=config['opt']
        )

    # length_of_film = 0
    if os.path.isfile(config['source']): 
        data = cv2.VideoCapture(app.source)
        # length_of_film = data.get(cv2.CAP_PROP_FRAME_COUNT)
        data.release()
        cv2.destroyAllWindows()

    with StepContext(name='Setup Logic',catch=(KeyError,IndexError)):
        app.setup_logic_module(args) 

    with StepContext(name='Setup MQTT', catch=(ConnectionError, TimeoutError)): 
        if app.mqtt:
            app.setup_mqtt(
                topic="test/topic", 
                broker_address="mqtt.eclipseprojects.io", 
                port=1883
            )
        else:
            logger.debug("[MQTT] interface is disabled") 

    with StepContext(name='Run_App', catch=(RuntimeError,)): 
    # Simulate publishing messages in intervals
         app.run_app()
         app.close_app()

   
if __name__ == "__main__":
    main()
    

