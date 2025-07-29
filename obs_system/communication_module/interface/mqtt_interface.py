from obs_system.utils.logger import logger
from abc import ABC, abstractmethod


class MQTTInterface(ABC):

    def __init__(self, broker_address, topic):
        self.broker_address = broker_address #MQTT host
        self.topic = topic


    @abstractmethod
    def connect(self,port,keepalive):
        pass


    @abstractmethod
    def publish(self, topic, message):
        pass


    @abstractmethod
    def subscribe(self, topic):
        pass


    # You can keep this method concrete if its behavior doesn't vary across implementations
    def on_message(self, client, userdata, message):
        logger.debug(f"Received message: {message.payload.decode()} on topic {message.topic}")


    def on_connect(self, client, userdata, flags, rc):
        
        if rc == 0:
            logger.debug(f"Connected with results code {rc} to the broker")
            client.subscribe(self.topic)
            logger.debug(f"Subscribing to topic {self.topic}")
        else: 
            logger.debug("Failed to connect, return code %d\n", rc)