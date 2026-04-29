import os 
import paho.mqtt.client as mqtt 
from pathlib import Path

JETSON_DEVICE_ID = 2
BROKER = f"edgejet{JETSON_DEVICE_ID}vpn.edi.lv"
PORT = 8884 
TOPIC = "reid-vehicle-detection" 
MESSAGE = "" 
CALLBACK_API_VERSION = mqtt.CallbackAPIVersion.VERSION2 
PROTOCOL = mqtt.MQTTv311 
QOS = 1 
TIMEOUT_S = 10 

CLIENT_NAME="edgeai-vcd42-hua"

CLIENT_CRT = f"assets/mqtt_credentials/jetson{JETSON_DEVICE_ID}/edgeai-vcd42-hua-edgejet{JETSON_DEVICE_ID}vpn/client-certs/client.crt" 
CLIENT_KEY = f"assets/mqtt_credentials/jetson{JETSON_DEVICE_ID}/edgeai-vcd42-hua-edgejet{JETSON_DEVICE_ID}vpn/client-certs/client.key" 
CA_CRT = f"assets/mqtt_credentials/jetson{JETSON_DEVICE_ID}/edgeai-vcd42-hua-edgejet{JETSON_DEVICE_ID}vpn/ca-cert" 

JPEG_QUALITY = 75 
KEEPALIVE = 60 

CREATE_SUBSCRIBER = True

MQTT_DIR = Path("./assets/mqtt") 
os.makedirs(MQTT_DIR, exist_ok=True)

SAVE_PUBLISHES_PATH = Path("./assets/mqtt/saved_publishes.cbor")
if not os.path.exists(SAVE_PUBLISHES_PATH): 
    with open(SAVE_PUBLISHES_PATH, "ab") as f : 
        f.write(b"\n") 

DECODED_JPEG_DIR = Path("./assets/mqtt/received_vehicle_crops")
os.makedirs(DECODED_JPEG_DIR, exist_ok=True)
