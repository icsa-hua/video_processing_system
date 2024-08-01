from obs_system.communication_module.interface.mqtt_interface import MQTTInterface
import paho.mqtt.client as mqtt

class RealMQTT(MQTTInterface):
    def __init__(self, broker_address):
        super().__init__(broker_address)
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
    
    def connect(self,port,keepalive):
        self.client.connect(self.broker_address)

    def publish(self, topic, message):
        self.client.publish(topic, message)

    def subscribe(self, topic):
        self.client.subscribe(topic)