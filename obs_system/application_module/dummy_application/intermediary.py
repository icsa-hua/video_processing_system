from __future__ import annotations

from obs_system.utils.common import find_available_port
from obs_system.utils.logger import get_logger

import signal
import socket
import subprocess
import sys
import time


logger = get_logger(name="obs_system." + __name__)


def gui_connector(host_address: str = "localhost", port_address: int = 8503):
    if host_address != "localhost" and host_address != "0.0.0.0":
        raise ValueError("Set host address to 'localhost'")

    host = socket.gethostbyname(host_address)
    port = find_available_port(port_address)
    if port is None:
        raise RuntimeError("No available port found for the Streamlit interface")

    fastapi_process: subprocess.Popen | None = None
    streamlit_process: subprocess.Popen | None = None

    def shutdown_handler(signum=None, frame=None):
        logger.debug("-- Received termination signal. Shutting down GUI services --")

        for proc in (streamlit_process, fastapi_process):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

        logger.debug("--- Shutdown complete ---")
        sys.exit(0)

    logger.debug(f"-- GUI will be running on http://{host}:{port} --")
    logger.debug(f"-- FastAPI will be running on http://{host}:8000/docs --")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        fastapi_process = subprocess.Popen(
            [
                "uvicorn",
                "obs_system.application_module.dummy_application.backend:server",
                "--host",
                str(host),
                "--port",
                "8000",
                "--loop", 
                "asyncio"
            ]
        )
        streamlit_process = subprocess.Popen(
            [
                "streamlit",
                "run",
                "obs_system/application_module/dummy_application/web_interface.py",
                f"--server.port={port}",
                f"--server.address={host}",
            ]
        )

        logger.debug("-- GUI is running --")

        while True:
            fastapi_code = fastapi_process.poll()
            streamlit_code = streamlit_process.poll()
            if fastapi_code is not None or streamlit_code is not None:
                if fastapi_code not in (None, 0):
                    logger.error("FastAPI exited with code %s", fastapi_code)
                if streamlit_code not in (None, 0):
                    logger.error("Streamlit exited with code %s", streamlit_code)
                shutdown_handler()
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_handler()
    except Exception as exc:
        logger.exception("-- Error starting GUI: %s --", exc)
        shutdown_handler()
