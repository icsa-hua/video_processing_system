from obs_system.utils.common import find_available_port
from obs_system.utils.logger import get_logger

import sys
import signal 
import socket 
import subprocess
import multiprocessing

from multiprocessing import Process

logger = get_logger(name="obs_system." + __name__)

def gui_connector(host_address:str="localhost", port_address:int=8503):

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
    

    if host_address != "localhost":
        raise ValueError("Set host address to 'Localhost'")

    host = socket.gethostbyname(host_address)
    port = find_available_port(port_address)
    
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
            logger.debug(f"Starting fastapi in {str(host)}...") 

        def start_streamlit():
            subprocess.run([
                "streamlit",
                "run",
                'obs_system/application_module/dummy_application/web_interface.py',
                f'--server.port={str(port)}',
                f"--server.address={str(host)}"
            ])
            logger.debug(f"Starting streamlit server in {str(host)}/{str(port)}")
        
    except Exception as e: 
        logger.exception(f"-- Error starting GUI: {e} --")
        sys.exit(1)

    except KeyboardInterrupt as keyboard_interrupt:
            logger.exception("-- KeyboardInterrupt: {} --".format(keyboard_interrupt))
            sys.exit(0)

    # Create the two processes
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



