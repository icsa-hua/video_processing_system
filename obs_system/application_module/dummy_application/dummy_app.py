from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QPixmap, QImage
from obs_system.conversion_module.dummy_convertor.frame_convertor import FrameConvertor
from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.detection_module.dummy_predictor.dummy_yolo import DummyPredictor
from obs_system.detection_module.dummy_predictor.stream_yolov5 import Yolov5Streamer
from obs_system.detection_module.dummy_predictor.stream_yolov8 import Yolov8Streamer
from obs_system.communication_module.mqtt_com.message_transmitter import RealMQTT
from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer


from obs_system.application_module.dummy_application.worker import ImageUpdateSignal
import numpy as np 
import warnings
import cv2
import os 
import psutil 
import pynvml
import time 
import multiprocessing
import pdb

class MainWindow(QMainWindow):
    """
    The `MainWindow` class is the main window of the application,
    which displays the image with bounding boxes detected by the object detector.
    
    ***Under Construction***
    
    """
        
    def __init__(self, object_detector):
        super().__init__()
        self.initUI()
        self.object_detector = object_detector
        self.imageUpdateSignal = ImageUpdateSignal()
        self.imageUpdateSignal.signal.connect(self.updateImageDisplay)
    
    def initUI(self):
        # Set up the main window
        self.setWindowTitle('YOLO Bounding Boxes Viewer')
        self.setGeometry(100, 100, 800, 600)
        self.label = QLabel("Image will be shown here")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        # Set up layout and widgets
        self.label = QLabel("Image will be shown here")
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def start_detection(self, patches) -> np.ndarray:
        predictions = self.object_detector.concurrent_prediction(patches)
        processed_patches = [self.object_detector.draw_boxes(patch, pred) for patch, pred in zip(patches, predictions)]
        for processed_patch in processed_patches:
            self.imageUpdateSignal.signal.emit(processed_patch)

    @pyqtSlot(np.ndarray)
    def updateImageDisplay(self, image:np.ndarray):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)

    
class Application: 
   
   
    def __init__(self):
        self.object_detector = None
        self.frame_counter = 0 
        self.model = None
        self.parent_path =  os.getcwd()
        
        # main_window = MainWindow(model_predictor)
        # main_window.show()


    def get_object_detector(self, model_name):
    
        set_object_detector_func = {
            "yolov5":self.yolov5_object_detector, 
            "YOLO5":self.yolov5_object_detector,
            "Yolo5":self.yolov5_object_detector, 
            "Yolov5":self.yolov5_object_detector,
            "yolov5s":self.yolov5_object_detector,
            "yolov8":self.yolov8_object_detector, 
            "YOLO8":self.yolov8_object_detector,
            "Yolo8":self.yolov8_object_detector, 
            "Yolov8":self.yolov8_object_detector,
            "yolov8s":self.yolov8_object_detector,
            "maskrcnn" :self.maskrcnn_object_detector,
            "MaskRCNN":self.maskrcnn_object_detector,
            "onnx":self.onnx_object_detector,
            "ONNX":self.onnx_object_detector
        } 
        return set_object_detector_func.get(model_name, lambda *args:None)


    def get_streaming_detector(self, model_name):
        
        set_object_detector_func = {
            "yolov5":self.yolov5_streaming,
            "YOLO5":self.yolov5_streaming,
            "Yolo5":self.yolov5_streaming,
            "Yolov5":self.yolov5_streaming,
            "yolov5s":self.yolov5_streaming,
            "yolov8":self.yolov8_streaming,
            "YOLO8":self.yolov8_streaming,
            "Yolo8":self.yolov8_streaming,
            "Yolov8":self.yolov8_streaming,
            "yolov8s":self.yolov8_streaming,
        }

        return set_object_detector_func.get(model_name, lambda *args:None)


    def yolov5_object_detector(self, model_name):
        model_weights = os.path.join(self.parent_path,'yolov5s.pt')
        self.model =  DummyPredictor(model_path=model_weights, model_name=model_name)


    def yolov8_object_detector(self, model_name):
        model_weights = os.path.join(self.parent_path,'yolov8n.pt')
        self.model =  DummyPredictor(model_path=model_weights, model_name=model_name)


    def maskrcnn_object_detector(self, model_name):
        model_weights = os.path.join(self.parent_path,'obs_system/Mask_RCNN/mask_rcnn_coco.h5')
        self.model =  DummyPredictor(model_path=model_weights, model_name=model_name)


    def onnx_object_detector(self, model_name):
        model_weights = os.path.join(self.parent_path,'obs_system/application_module/dummy_application/onnx_workflow/dummy_compression/yolov5s.onnx')
        self.model =  DummyPredictor(model_path=model_weights, model_name=model_name)


    def yolov5_streaming(self, opt):
        from ultralytics.utils import DEFAULT_CFG
        model_weights = 'yolov5s.pt'
        self.streamer = Yolov5Streamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model=model_weights, verbose=False, opt=opt)
        self.model = self.streamer.model


    def yolov8_streaming(self, opt):
        from ultralytics.utils import DEFAULT_CFG
        model_weights = 'yolov8n.pt'
        self.streamer = Yolov8Streamer(DEFAULT_CFG, {}, None)
        self.streamer.setup_model(model=model_weights, verbose=False, opt=opt)
        self.model = self.streamer.model


    def setup_process(self, source): 
        self.process_memory = psutil.Process(os.getpid())
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        parent_conn, child_conn = multiprocessing.Pipe()    
        self.source = os.path.join(self.parent_path, source) if os.path.isfile(source) else source
        #Create the data acquisition from video source 
        consumer = DummyConsumer(child_conn, self.source)
        self.process = multiprocessing.Process(target=consumer.run, args=())
        self.process.start()    
        self.start_time = time.time()
        return parent_conn


    def setup_model(self, model_name, stream, opt="tracking"):
        self.stream = stream
        if stream: 
            setup_func = self.get_streaming_detector(model_name)
            return setup_func(opt=opt)
        else:
            setup_func = self.get_object_detector(model_name)
            return setup_func(model_name=model_name)


    def setup_logic_module(self): 
        self.logic_module = BoundingBoxOverlapDetector()


    def run_app(self, parent_conn, output_path, save, model, length_of_film=0 ): 
        process_video_func = self.get_process_video()
        return process_video_func(parent_conn=parent_conn, output_path=output_path, save=save, model=model, length_of_film=length_of_film)


    def get_process_video(self): 
        
        if self.stream :
            return  self.process_stream
          
        return  self.process_video

 
    

    def process_stream(self, parent_conn, output_path, save, model, length_of_film=0):
        
        if isinstance(self.source, str):
            # length_of_film = self.get_length_of_film(self.source)
            #Streaming the video as before 
            print("Process Streaming operation")
            
            self.streamer(source=self.source, model=model, stream=self.stream, mqtt_broker=self.mqtt_interface) # e(source=0, model=model_weights, stream=True)                
            return self.streamer.results
        



        #Batching the video source. 
        return []



    def process_video(self, parent_conn, output_path, save, model, length_of_film=0):
        while True:   
            if parent_conn.poll():
                try : 

                    try: 
                        frame, shape = self.get_image_from_process(parent_conn)
                    except:
                            warnings.warn("No frame received from the video_source.")
                            break 
                    
                    if self.frame_counter > length_of_film and length_of_film!=0: 
                        break

                    convertor = FrameConvertor(frame=frame, shape=shape, output_path=output_path, save_conversion=save, model_name=model)
                    patches, positions = convertor.decomposition()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        results = [] 
                        if patches is not None:
                            predictions = self.object_detector.concurrent_prediction(patches)
                            full_image = convertor.reconstruct_image(patches, positions)
                            stride = convertor.get_stride() 

                            predictions = self.object_detector.draw_boxes(full_image, patches, positions, predictions, shape, stride)
                            full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR) 
                            # cv2.imshow('Detections on large image', full_image)
                            # cv2.waitKey(100)    

                            # #Detect overlapping boxes 
                            results = self.logic_module.detect(predictions)
                            overlap_frame = self.logic_module.draw_overlaps(full_image, np.asarray(results))

                            #Show image 
                            cv2.imshow("Overlap frame", overlap_frame)
                            cv2.waitKey(1)

                            if self.frame_counter<=20: 
                                
                                directory = r'obs_system/converted_mp4/det/images/'
                                os.chdir(directory)
                                filename = f"obs_frame_{self.frame_counter}.png"
                                cv2.imwrite(filename,full_image)
                        
                            self.frame_counter+=1

                        if not results: 
                            self.publish_mqtt("No object detected")
                        else: 
                            self.publish_mqtt(str(results))           
                        
                except Exception as e:
                    print("Error while receiving/processing frame:", e)
        
        raise KeyboardInterrupt("Mannually triggered exception")

            
            

    def get_image_from_process(self, process): 
        shape, dtype, data = process.recv()
        frame = np.frombuffer(data, dtype=dtype)
        frame = frame.reshape(shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, shape


    def setup_mqtt(self, topic, broker_address, port):
        self.mqtt_topic = topic
        self.mqtt_interface = RealMQTT(broker_address, self.mqtt_topic)
        self.mqtt_interface.connect(port=port, keepalive=60)
        self.mqtt_interface.client.loop_start() #Not loop.forever as main thread will be taken over for the MQTT process. 


    def publish_mqtt(self, message):
        self.mqtt_interface.publish(topic=self.mqtt_topic, message=message)


    def statistics(self):
        end_time = time.time()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        total_gpu_mem = mem_info.total / (1024 ** 2)  # Convert bytes to MB
        used_gpu_mem = mem_info.used / (1024 ** 2)
        free_gpu_mem = mem_info.free / (1024 ** 2)
        print(f"-------------------------------------")
        print(f"|  Total Inference time: {end_time - self.start_time} seconds ")
        print(f"|  Current RAM usage: {self.process_memory.memory_info().rss / (1024**2)} MB.")
        print(f"|  Total GPU memory: {total_gpu_mem:.2f} MB")
        print(f"|  Used GPU memory: {used_gpu_mem:.2f} MB")
        print(f"|  Free GPU memory: {free_gpu_mem:.2f} MB")
        print(f"-------------------------------------")


    def close_app(self): 

        #Display GPU usage after execution
        self.statistics()

        # #Close MQTT connection with server
        self.mqtt_interface.client.loop_stop()
        self.mqtt_interface.client.disconnect()

        self.process.terminate()
        self.process.join()

        pynvml.nvmlShutdown()


