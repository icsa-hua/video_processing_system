from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.utils.logger import logger
from obs_system.utils.common import *

import os
import cv2
import sys
import signal 
import socket 
import argparse
import subprocess
import multiprocessing

from multiprocessing import Process
from silence_tensorflow import silence_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #suppress warnings. 
silence_tensorflow()


def main():
    logger.info("--- Initializing Application ---")
    
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--model_name', metavar='M', default='onnx', help='Model to use (Yolov5, Yolov8 (Default), MaskRCNN, ONNX (yolov5, yolov8))')
    argparser.add_argument('--source', metavar='SO', default='samples/sample_video.mp4', help='Source to use - Local video path (.mp4) or stream index (key needs to be provided)')
    argparser.add_argument('--type', metavar='T', default='tracking', help='Use tracking with bytetracker or simple detection (recommended to leave default value)')
    argparser.add_argument('--gui', metavar='G', action=argparse.BooleanOptionalAction, help='Use GUI to select video source and model')
    argparser.add_argument('--mqtt',metavar='M', action=argparse.BooleanOptionalAction, help='Use MQTT to send data to server')
    argparser.add_argument('--show', metavar='SH', action=argparse.BooleanOptionalAction, help='Show real-time result inference')
    argparser.add_argument('--verbose', metavar='V', action=argparse.BooleanOptionalAction, help='Show results of inference in stdout')
    argparser.add_argument('--port', metavar='P', default=8503, help='Port for Streamlit service interface')
    argparser.add_argument('--host_server', metavar='H' , default= 'localhost', help='Host server for both streamlit and fastapi')
    argparser.add_argument('--save', metavar='SA', action=argparse.BooleanOptionalAction, help='Save inference results to file')
    argparser.add_argument('--DAV2', metavar='D', action=argparse.BooleanOptionalAction, help='Use depth imaging to detect obstacles') 
    argparser.add_argument('--roi', metavar='R', action=argparse.BooleanOptionalAction, help='Use Region of Interest to detect obstacles')

    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()

    if args.gui: 
        
        if sys.platform.startswith("win"):
            multiprocessing.set_start_method("spawn", force=True)
        elif sys.platform.startswith("linux"):
            multiprocessing.set_start_method("fork", force=True)

        global fastapi_process, streamlit_process
        
        def shutdown_handler(signum, frame):
            global fastapi_process, streamlit_process
            
            logger.debug("--Received termination signal. Shutting down... --")

            if streamlit_process and streamlit_process.is_alive(): 
                streamlit_process.terminate()
                streamlit_process.join()
                streamlit_process.close()

            if fastapi_process and fastapi_process.is_alive():
                fastapi_process.terminate()
                fastapi_process.join()
                fastapi_process.close() 

            logger.debug("--- Shutdown complete ---")
            sys.exit(0)

        host  = args.host_server

        if args.host_server == "localhost": 
            host = socket.gethostbyname("localhost")
            
        port = find_available_port(args.port)

        logger.debug(f"-- GUI will be running on http://{str(host)}:{port} --")
        logger.debug(f"-- FastAPI will be running on http://{str(host)}:8000/docs --")

        try: 
            def start_fastapi():
                subprocess.run([
                    "fastapi",
                    "dev",
                    "obs_system/application_module/dummy_application/backend.py", 
                    '--reload', 
                    '--host', str(host)
                ])

            def start_streamlit():
                subprocess.run(["streamlit", "run", 'obs_system/application_module/dummy_application/web_interface.py', f'--server.port={str(port)}', f"--server.address={str(host)}"])
        
        except Exception as e:
            logger.exception(f"-- Error starting GUI: {e} --")
            sys.exit(1)

        except KeyboardInterrupt as keyboard_interrupt:
            logger.exception("-- KeyboardInterrupt: {} --".format(keyboard_interrupt))
            sys.exit(0)
        
        try:
            fastapi_process = Process(target=start_fastapi)
            streamlit_process = Process(target=start_streamlit)

            # Start processes
            fastapi_process.start()
            streamlit_process.start()
            logger.debug("-- ✅ GUI is running --")

            # Catch termination signals 
            signal.signal(signal.SIGINT, shutdown_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, shutdown_handler)  # Kill command

            fastapi_process.join()
            streamlit_process.join()   
            logger.debug("-- GUI terminated --")

        except KeyboardInterrupt as keyboard_interrupt:
            logger.exception("-- ❌ KeyboardInterrupt: {} --".format(keyboard_interrupt))
            shutdown_handler(None ,None)
            
        except Exception as e:
            logger.exception(f"-- Error starting GUI: {e} --")
            shutdown_handler(None ,None)

        return 
    
    config = {
        'model_name':args.model_name,
        'stream':True, 
        'source':args.source, 
        'model_type':args.type, 
        'save':args.save, 
        'verbose':args.verbose,
    }   

    model_key = config['model_name'].lower()

    # Check that the model name responds to the models approved for this application (Yolov5-v8) 
    config['model_name'] = check_model_name(model_key=model_key, condition= config['model_type'], condition_type=args.type)

    logger.info("-- Initial Configuration Complete --\n -- ✅ Starting Application --")
    
    #Initialize the application module that interfaces source, model, mqtt and logic module 
    app = Application()
    app.setup_process(config['source'], args)
    try: 
        app.setup_model(
            model_name=model_key,
             stream=config['stream'], 
            opt=str(config['model_type'])
        )
        logger.debug("-- Model Initialized --") 

    except Exception as e:
        logger.exception(e)
        exit(1)

    if app.model is None: 
        logger.error("-- model not initialized correctly. shutting down --")
        raise exception("-- ❌ model not initialized. shutting down --")
    
    
    # length_of_film = 0
    if os.path.isfile(config['source']): 
        data = cv2.VideoCapture(app.source)
        # length_of_film = data.get(cv2.CAP_PROP_FRAME_COUNT)
        data.release()
        cv2.destroyAllWindows()
    
    try:
        app.setup_logic_module(args)
        logger.info("-- Logic module initialized --")
    except Exception as e:
        logger.exception(f"-- Error setting up logic module: {e} --")
        exit(1)
    
    try:
        if app.mqtt:
            app.setup_mqtt(topic="test/topic",
                        broker_address="mqtt.eclipseprojects.io",
                        port=1883)
            logger.debug("-- MQTT interface connected --")
        else: 
            logger.debug("-- MQTT interface not enabled --")
            
    except Exception as e:
        logger.exception(f"-- Error setting up MQTT: {e} --")
        exit(1)
    
    # Simulate publishing messages in intervals
    try:
        app.run_app(model=config['model_name'])
        app.close_app()

    except KeyboardInterrupt as e:
        logger.exception(f"-- Exception caught: {e} --")
        app.close_app()
        

if __name__ == "__main__":
    main()
    

