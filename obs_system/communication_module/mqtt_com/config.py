import os 
import paho.mqtt.client as mqtt 
from pathlib import Path

BROKER = "localhost" 
PORT = 1883 
TOPIC = "test/topic" 
MESSAGE = "" 
CALLBACK_API_VERSION = mqtt.CallbackAPIVersion.VERSION2 
PROTOCOL = mqtt.MQTTv311 
QOS = 1 
TIMEOUT_S = 10 

CLIENT_CRT = "path/to/client.crt" 
CLIENT_KEY = "path/to/client.key" 
CA_CRT = "path/to/ca.crt" 

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
