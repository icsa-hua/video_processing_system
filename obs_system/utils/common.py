import subprocess
import socket

# check the existence of GPU 
def check_nvidia_existence(): 

    try: 
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return result.returncode == 0 
    
    except ValueError:
        return False
    


def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts): 
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.settimeout(1) 
            host = socket.gethostbyname("localhost")
            if s.connect_ex((host, port)) != 0: 
                return port 
            
    return None 



def check_model_name(model_key:str,condition:str,condition_type:str): 

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
        'yolov8m': ('autobackbone', 'y8'),
        'onnx' : ('compressed', 'y8'), 
        'compressed' : ('compressed', 'y8') 
    }
    if model_key in model_validation and condition==condition_type:
            return model_validation[model_key][0]
    else: 
        raise ValueError("No valid model was provided...\nUse 'yolov8' as an example")
