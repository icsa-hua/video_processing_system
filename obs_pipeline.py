from obs_system.application_module.dummy_application.dummy_app import Application

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #suppress warnings. 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import cv2
import sys
import signal 
import argparse
import logging
import socket 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts): 
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.settimeout(1) 
            host = socket.gethostbyname("localhost")
            if s.connect_ex((host, port)) != 0: 
                return port 
            
    return None 



def main():
    logger.info("Getting Initial Configuration...")
    
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--name', metavar='M', default='yolov8', help='Model to use (Yolov5, Yolov8 (Default), MaskRCNN, ONNX (yolov5, yolov8))')
    argparser.add_argument('--source', metavar='S', default='samples/sample_video.mp4', help='Source to use - Local video path (.mp4) or stream index (key needs to be provided)')
    argparser.add_argument('--type', metavar='T', default='tracking', help='Use tracking with bytetracker or simple detection (recommended to leave default value)')
    argparser.add_argument('--gui', metavar='G', action=argparse.BooleanOptionalAction, help='Use GUI to select video source and model')
    argparser.add_argument('--mqtt',metavar='M', action=argparse.BooleanOptionalAction, help='Use MQTT to send data to server')
    argparser.add_argument('--show', metavar='H', action=argparse.BooleanOptionalAction, help='Show real-time result inference')
    argparser.add_argument('--verbose', metavar='V', action=argparse.BooleanOptionalAction, help='Show results of inference in stdout')
    argparser.add_argument('--port', metavar='P', default=8503, help='Port for Streamlit service interface')
    argparser.add_argument('--host_server', metavar='H' , default= 'localhost', help='Host server for both streamlit and fastapi')
    argparser.add_argument('--save', metavar='S', action=argparse.BooleanOptionalAction, help='Save inference results to file')


    if len(sys.argv) < 1:
         argparser.print_help()
         return
    
    args = argparser.parse_args()

    if args.gui: 
        import subprocess
        import multiprocessing
        from multiprocessing import Process

        if sys.platform.startswith("win"):
            multiprocessing.set_start_method("spawn", force=True)
        elif sys.platform.startswith("linux"):
            multiprocessing.set_start_method("fork", force=True)

        global fastapi_process, streamlit_process
        def shutdown_handler(signum, frame):
            global fastapi_process, streamlit_process
            
            logger.info("Received termination signal. Shutting down...")

            if streamlit_process and streamlit_process.is_alive(): 
                streamlit_process.terminate()
                streamlit_process.join()
                streamlit_process.close()

            if fastapi_process and fastapi_process.is_alive():
                fastapi_process.terminate()
                fastapi_process.join()
                fastapi_process.close() 

            logger.info("Shutdown complete.")
            sys.exit(0)
        host  = args.host_server
        if args.host_server == "localhost": 
            host = socket.gethostbyname("localhost")
        port = find_available_port(args.port)
        logger.info(f"GUI will be running on http://{host}:{port}")
        logger.info(f"FastAPI will be running on http://{str(host)}:8000/docs")
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
            logger.error(f"Error starting GUI: {e}")
            sys.exit(1)
        except KeyboardInterrupt as keyboard_interrupt:
            logger.info("KeyboardInterrupt: {}".format(keyboard_interrupt))
            sys.exit(0)
        

        try:
            fastapi_process = Process(target=start_fastapi)
            streamlit_process = Process(target=start_streamlit)

            # Start processes
            fastapi_process.start()
            streamlit_process.start()
            logger.info("GUI is running...")

            # Catch termination signals 
            signal.signal(signal.SIGINT, shutdown_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, shutdown_handler)  # Kill command

            fastapi_process.join()
            streamlit_process.join()   
            logger.info("GUI terminated...")

        except KeyboardInterrupt as keyboard_interrupt:
            logger.info("KeyboardInterrupt: {}".format(keyboard_interrupt))
            shutdown_handler(None ,None)
        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
            shutdown_handler(None ,None)

        return 
    

    config = {
        'model_name':args.name,
        'stream':True, 
        'source':args.source, 
        'model_type':args.type, 
        'save':args.save, 
        'verbose':args.verbose,
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
    app = Application(logger)
    app.setup_process(config['source'], args)

    try: 
        app.setup_model(model_name=model_key, 
                        stream=config['stream'],
                        opt=config['model_type'])
        logger.info("Model Initialized...") 

    except Exception as e:
        logger.error(e)
        exit(1)

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
    

    app.setup_logic_module()
    
    try:
        if app.mqtt:
            app.setup_mqtt(topic="test/topic",
                        broker_address="mqtt.eclipseprojects.io",
                        port=1883)
            logger.info("MQTT interface connected...")
        else: 
            logger.info("MQTT interface not enabled.")
            
    except Exception as e:
        logger.error(f"Error setting up MQTT: {e}")
        exit(1)
    
    # Simulate publishing messages in intervals
    try:
        app.run_app(output_path=converted_video_path, 
                    save=False, 
                    model=config['model_name'], 
                    length_of_film=length_of_film)
        app.close_app()

    except KeyboardInterrupt as e:
        logger.error(f"Exception caught: {e}")
        logger.error("Terminating application...")
        app.close_app()
        


if __name__ == "__main__":
    main()
    

