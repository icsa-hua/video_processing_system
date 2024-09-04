from abc import ABC, abstractmethod


class MQTTInterface(ABC):
    def __init__(self, broker_address, topic):
        self.broker_address = broker_address #MQTT host
        self.topic = topic


    @abstractmethod
    def connect(self):
        pass


    @abstractmethod
    def publish(self, topic, message):
        pass


    @abstractmethod
    def subscribe(self, topic):
        pass


    # You can keep this method concrete if its behavior doesn't vary across implementations
    def on_message(self, client, userdata, message):
        print(f"Received message: {message.payload.decode()} on topic {message.topic}")


    def on_connect(self, client, userdata, flags, rc):
        
        if rc == 0:
            print(f"Connected with results code {rc} to the broker")
            client.subscribe(self.topic)
            print(f"Subscribing to topic {self.topic}")
        else: 
            print("Failed to connect, return code %d\n", rc)