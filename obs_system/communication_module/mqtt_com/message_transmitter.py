from obs_system.communication_module.interface.mqtt_interface import MQTTInterface
from obs_system.communication_module.mqtt_com.config import CA_CRT, CLIENT_CRT, CLIENT_KEY
from obs_system.utils.logger import get_logger 
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Callable, Union, List, Sequence
from obs_system.communication_module.mqtt_com.config import * 

import os 
import cv2
import ssl 
import time
import cbor2
import numpy as np
import paho.mqtt.client as mqtt

BBoxXYWH = Tuple[int, int, int, int]
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


@dataclass 
class CroppedItem: 
    cls: str 
    conf: float 
    img: bytes 
    w: int 
    h: int 
    fmt: str = "jpg" 
    track_id: Optional[int] = None 

    # BBox in original frame corrds
    bbox: Optional[BBoxXYWH] = None 


@dataclass
class CropBatchMessage:
    """Convenience structure for decoded messages."""
    v: int
    ts: int
    cam: str
    items: List[CroppedItem] 
    frame_id: Optional[int] = None 


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CropBatchMessage":
        items: List[CroppedItem] = [] 

        for it in d.get("items", []): 
            items.append(
                CroppedItem(
                    cls=str(it.get("cls", "unkown")), 
                    conf=float(it.get("conf", 0.0)), 
                    img=bytes(it.get("img")), 
                    w=int(it.get("w", 0)), 
                    h=int(it.get("h", 0)), 
                    fmt=str(it.get("fmt", "jpg")), 
                    track_id=int(it["track_id"]) if "track_id" in it and it["track_id"] is not None else None, 
                    bbox=tuple(map(int, it["box"])) if "bbox" in it and it["bbox"] is not None else None 
                )
            ) 

        return CropBatchMessage(
            v=int(d.get("v", 1)), 
            ts=int(d["ts"]), 
            cam=str(d["cam"]), 
            items=items, 
            frame_id=int(d["frame_id"]) if "frame_id" in d and d["frame_id"] is not None else None 
        )


class CBORMQTTCropClientCV2(MQTTInterface):
    """
    MQTT client for CBOR-wrapped JPEG crops using OpenCV.

    Supports:
      - single crop messages
      - batch crop messages (recommended when many vehicles per frame)
   
    """

    def __init__(
        self,
        broker_address: str,
        topic: str,
        callback_version: Any, 
        client_id: str = "cbor-crop-client",
        qos: int = 0,
        jpeg_quality: int = 80,
    ):
        super().__init__(broker_address, topic)

        self.qos = int(qos)
        self.jpeg_quality = int(jpeg_quality)
        self.broker_address = broker_address 
        self.topic = topic

        # Paho MQTT client (v2 callback API)
        self.client = mqtt.Client(callback_version, client_id=client_id)
        # self.client.on_connect = self.on_connect
        # self.client.on_message = self.on_message

        # Optional pipeline callback invoked with decoded messages
        self._on_batch_callback: Optional[Callable[[CropBatchMessage, str], None]] = None

        if 'sender' in client_id and (os.path.exists(CLIENT_CRT) and os.path.exists(CLIENT_KEY) and os.path.exists(CA_CRT)): 
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_verify_locations(CA_CRT) 
            ctx.load_cert_chain(CLIENT_CRT, CLIENT_KEY) 
            self.client.tls_set_context(ctx) 


    # ---- Public helpers ----
    def set_on_batch_callback(self, cb: Callable[[CropBatchMessage, str], None]) -> None:
        self._on_batch_callback = cb


    def start_loop(self, background: bool = True) -> None:
        if background:
            self.client.loop_start()
        else:
            self.client.loop_forever()


    def stop_loop(self) -> None:
        self.client.loop_stop()


    # ---- MQTTInterface abstract methods ----

    def connect(self, port: int = 1883, keepalive: int = 60):
        self.client.connect(self.broker_address, int(port), int(keepalive))
        logger.debug(f"Connecting to broker {self.broker_address}:{port} keepalive={keepalive}")


    def publish(self, topic: str, message: Union[bytes, Dict[str, Any]]):
        payload = cbor2.dumps(message) if isinstance(message, dict) else message
        info = self.client.publish(topic, payload=payload, qos=self.qos, retain=False)
        return info


    def subscribe(self, topic: str):
        self.client.subscribe(topic, qos=self.qos)
        logger.debug(f"Subscribing to topic {topic} qos={self.qos}")


    # ---- Encoding utilities (OpenCV) ----

    def _jpeg_encode(self, bgr: np.ndarray) -> Tuple[bytes, int, int]:
        if bgr is None or bgr.size == 0: 
            raise ValueError("Empty image provided for encoding") 

        if bgr.ndim != 3 or bgr.shape[2] != 3: 
            raise ValueError(f"Expected BGR image shape (H, W, 3), got {bgr.shape}") 

        h, w = int(bgr.shape[0]), int(bgr.shape[1]) 
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)] 
        ok, buf = cv2.imencode(".jpg", bgr, params) 
        if not ok : 
            raise RuntimeError("cv2.imencode('.jpg'), ...) failed") 

        return buf.tobytes(), w, h 


    def encode_batch_from_crops(
        self,
        crops: Sequence[Dict[str, Any]], 
        cam_id: str,
        ts_ms: Optional[int] = None,
        frame_id: Optional[int] = None, 
        include_bbox: bool = True, 
    ) -> bytes:
        """
        Build ONE CBOR payload containing multiple cropped objects.

        `crops` is your pipeline output list, where each item is expected to include:
          - "img": np.ndarray BGR crop (required)
          - "cls": str (optional, default 'unknown')
          - "conf": float (optional, default 0.0)
          - "track_id": int (optional)
          - "bbox": (x,y,w,h) in original frame coords (optional)

        include_bbox:
          - True -> include bbox if present
          - False -> omit bbox entirely (smaller payload)
        """


        items: List[Dict[str, Any]] = [] 
        for obj in crops: 
            crop_bgr = obj.get("img") 
            if crop_bgr is None: 
                continue 

            jpeg_bytes, w, h = self._jpeg_encode(crop_bgr) 

            it = {
                'cls':str(obj.get("cls", "unknown")), 
                'conf': float(obj.get("conf", 0.0)), 
                'fmt': "jpg", 
                'w': int(w), 
                'h': int(h), 
                "img": jpeg_bytes
            }

            if obj.get("track_id", None) is not None: 
                it["track_id"] = int(obj["track_id"]) 

            if include_bbox and obj.get("bbox", None) is not None: 
                x, y, bw, bh = obj["bbox"] 
                it["bbox"] = [int(x), int(y), int(bw), int(bh)]

            items.append(it) 

        msg: Dict[str, Any] = {
            "v": 1,
            "type": "crop_batch", 
            "ts": int(ts_ms if ts_ms is not None else time.time() * 1000),
            "cam": cam_id,
            "items": items, 
        }
        if frame_id is not None:
            msg["frame_id"] = int(frame_id)

        return cbor2.dumps(msg)


    def publish_batch_from_crops(
            self,
            crops: Sequence[Dict[str, Any]], 
            cam_id: str, 
            topic: Optional[str] = None, 
            ts_ms: Optional[int] = None, 
            frame_id: Optional[int] = None, 
            include_bbox: bool = True
    ):

        payload = self.encode_batch_from_crops(crops=crops, cam_id=cam_id, ts_ms=ts_ms, frame_id=frame_id, include_bbox=include_bbox)
        return self.publish(topic or self.topic, payload)

    
    # ---- Callbacks ----
    def on_message(self, client, userdata, message):
        try:
            data = cbor2.loads(message.payload)
            if isinstance(data, dict) and data.get("type") == "crop_batch": 
                batch = CropBatchMessage.from_dict(data) 

                logger.debug(
                    f"Received batch on {message.topic}: cam={batch.cam} ts={batch.ts} "
                    f"items={len(batch.items)} payload={len(message.payload)}B"
                )
                
                if self._on_batch_callback is not None: 
                    self._on_batch_callback(batch, message.topic) 

            else: 
                logger.debug(f"Received CBOR non-batch on {message.topic} keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        except Exception as e: 
            logger.debug(f"Received non-CBOR or malformed message on {message.topic} ({len(message.payload)} bytes): {e}")


    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.debug(f"Connected with results code {rc} to the broker")
            client.subscribe(self.topic, qos=self.qos)
            logger.debug(f"Subscribing to topic {self.topic} qos={self.qos}")
        else:
            logger.debug("Failed to connect, return code %d\n", rc)


def on_batch(batch: CropBatchMessage):
    os.makedirs(OUTDIR, exist_ok=True)
    for i, item in enumerate(batch.items):
        arr = np.frombuffer(item.img, dtype=np.uint8)
        crop_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            continue
        path = os.path.join(OUTDIR, f"{batch.cam}_{batch.ts}_{i}_{item.cls}_{item.track_id or 'na'}.jpg")
        cv2.imwrite(path, crop_bgr)

