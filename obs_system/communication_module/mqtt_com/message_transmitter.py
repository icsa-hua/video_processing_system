from obs_system.communication_module.interface.mqtt_interface import MQTTInterface
from obs_system.utils.logger import get_logger 

import paho.mqtt.client as mqtt


logger = get_logger("obs_system."+__name__)

class RealMQTT(MQTTInterface):

    def __init__(self, broker_address, topic):
        super().__init__(broker_address, topic) #Initializes the self.broker_address
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        
        
    def connect(self,port,keepalive):
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address, port, keepalive)
        logger.debug(f"Connecting to {self.broker_address} ")
        

    def publish(self, topic, message):
        self.client.publish(topic, message)
        logger.debug(f"Publishing '{message}' to topic '{topic}' ")


    def subscribe(self, topic):
        self.client.subscribe(topic)
        logger.debug(f"Subscribing to topic '{topic}' ")
