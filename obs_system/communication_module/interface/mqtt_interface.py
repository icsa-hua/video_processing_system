from abc import ABC, abstractmethod

class MQTTInterface(ABC):
    def __init__(self, broker_address):
        self.broker_address = broker_address

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
