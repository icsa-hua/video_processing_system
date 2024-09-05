from obs_system.communication_module.interface.mqtt_interface import MQTTInterface

class DummyMQTT(MQTTInterface):
    def connect(self):
        print(f"Connecting to {self.broker_address} (Dummy)")

    def publish(self, topic, message):
        print(f"Publishing '{message}' to topic '{topic}' (Dummy)")

    def subscribe(self, topic):
        print(f"Subscribing to topic '{topic}' (Dummy)")
 