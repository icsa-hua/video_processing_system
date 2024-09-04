from obs_system.communication_module.interface.mqtt_interface import MQTTInterface
import paho.mqtt.client as mqtt


class RealMQTT(MQTTInterface):


    def __init__(self, broker_address, topic):
        super().__init__(broker_address, topic) #Initializes the self.broker_address
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        
        
    def connect(self,port,keepalive):
        print(f"Connecting to {self.broker_address} ")
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address, port, keepalive)
        


    def publish(self, topic, message):
        print(f"Publishing '{message}' to topic '{topic}' ")

        self.client.publish(topic, message)


    def subscribe(self, topic):
        print(f"Subscribing to topic '{topic}' ")
        self.client.subscribe(topic)