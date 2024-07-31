from obs_system.detection_module.interface.predictor import ModelLoader
from obs_system.detection_module.interface.soft_nms import py_cpu_softnms

import numpy as np
import cv2 
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import uuid
import sys 
import time

# import supervision as sp
class DummyPredictor(ModelLoader):

    def __init__(self,model_path: str, model_name:str) -> None:
        
        self.model = None
        self.model_name = model_name

        #CoCo classes (mostly for YOLO)
        self.class_names = {0: u'person', 1: u'bicycle',2: u'car', 3: u'motorcycle', 
             4: u'airplane', 5: u'bus', 6: u'train', 7: u'truck', 8: u'boat', 9: u'traffic light', 
             10: u'fire hydrant', 11: u'stop sign', 12: u'parking meter', 13: u'bench', 14: u'bird', 
             15: u'cat', 16: u'dog', 17: u'horse', 18: u'sheep', 19: u'cow', 20: u'elephant', 
             21: u'bear', 22: u'zebra', 23: u'giraffe', 24: u'backpack', 25: u'umbrella', 26: u'handbag',
             27: u'tie', 28: u'suitcase', 29: u'frisbee', 30: u'skis', 31: u'snowboard', 32: u'sports ball', 
             33: u'kite', 34: u'baseball bat', 35: u'baseball glove', 36: u'skateboard', 37: u'surfboard', 
             38: u'tennis racket', 39: u'bottle', 40: u'wine glass', 41: u'cup', 42: u'fork', 43: u'knife', 
             44: u'spoon', 45: u'bowl', 46: u'banana', 47: u'apple', 48: u'sandwich', 49: u'orange', 
             50: u'broccoli', 51: u'carrot', 52: u'hot dog', 53: u'pizza', 54: u'donut', 55: u'cake', 
             56: u'chair', 57: u'couch', 58: u'potted plant', 59: u'bed', 60: u'dining table', 
             61: u'toilet', 62: u'tv', 63: u'laptop', 
             64: u'mouse', 65: u'remote', 66: u'keyboard', 67: u'cell phone', 68: u'microwave', 
             69: u'oven', 70: u'toaster', 71: u'sink', 72: u'refrigerator', 73: u'book', 74: u'clock', 
             75: u'vase', 76: u'scissors', 77: u'teddy bear', 78: u'hair drier', 79: u'toothbrush'}
        
        #Device to run model on
        self.device = None 
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.frame_id = 1
        self.load_model(model_path)
        print("Completed Model Configuration")
        
    def get_load_model_func(self, model_name:str): 
        load_model_func = {
            "yolo": self.load_yolo,
            "Yolo": self.load_yolo,
            "YOLO": self.load_yolo,
            "maskrcnn": self.load_maskrcnn,
            "MaskRCNN": self.load_maskrcnn,
            "Mask R-CNN": self.load_maskrcnn, 
            "onnx" : self.load_onnx, 
            "ONNX" : self.load_onnx,
            "Onnx" : self.load_onnx
        }
        return load_model_func.get(model_name, lambda *args:None)

    def get_predict_func(self,model_name:str):
        predict_func = {
            "yolo": self.predict_yolo,
            "Yolo": self.predict_yolo,
            "YOLO": self.predict_yolo,
            "maskrcnn": self.predict_maskrcnn,
            "MaskRCNN": self.predict_maskrcnn,
            "Mask R-CNN": self.predict_maskrcnn,
            "onnx" : self.predict_onnx,
            "ONNX" : self.predict_onnx,
            "Onnx" : self.predict_onnx
        }
        return predict_func.get(model_name, lambda *args: None)

    def get_draw_func(self,model_name:str):
        draw_func = {
            "yolo": self.draw_yolo_boxes,
            "Yolo": self.draw_yolo_boxes,
            "YOLO": self.draw_yolo_boxes,
            "maskrcnn": self.draw_maskrcnn_boxes,
            "MaskRCNN": self.draw_maskrcnn_boxes,
            "Mask R-CNN": self.draw_maskrcnn_boxes
        }
        return draw_func.get(model_name, lambda *args: None)


    def load_yolo(self, model_path: str) -> None:

        try:
            import torch
            cuda_use = torch.cuda.is_available()
            self.device = torch.device("cuda" if cuda_use else "cpu")
            if cuda_use :
                print("CUDA is available. GPU is working!")
            else:
                print("CUDA is not available. GPU is not working.")
        
            #Load Model 
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)
            self.model.conf = 0.25
            self.model.iou = 0.50
            self.model.multi_label = False
            self.model.max_det = 500

        except ImportError as e:
            print(e)
            print(sys.exc_type)

    def load_maskrcnn(self,model_path: str) -> None:
        
        try: 
            import obs_system.Mask_RCNN.mrcnn.model as modellib
            from obs_system.detection_module.interface.config import Camera360Config
            config = Camera360Config()
            config.display()
            self.model = modellib.MaskRCNN(mode='inference', model_dir="/obs_system/Mask_RCNN", config=config)
            self.model.load_weights(model_path, by_name=True)
       
        except ImportError as e: 
            print(e)
            print(sys.exc_type)

    def load_onnx(self, model_path:str) -> None: 
        try: 
            from obs_system.application_module.dummy_application.onnx_workflow.dummy_compression.compression import Compression
            
            compressor = Compression(model_path)
            compressor.quantize_model()
            # compressor.session_create()
            compressor.load_model(compressor.quantized_model_path)
            self.model = compressor
            print("Finished loading model")

        except ImportError as e:
            print(e)
            print(sys.exc_type)

    def load_model(self, model_path: str) -> None:
        load_model_func  = self.get_load_model_func(self.model_name)
        load_model_func(model_path)


    def predict_yolo(self,patch):
        try: 
            from datetime import datetime 
            results = self.model(patch)
            verbose = False
            if verbose: 
                results.save() #save images with detections. 
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                plt.imshow(np.squeeze(results.render()))
                plt.savefig(f"./patches_imgs/{str(uuid.uuid4().fields[-1])[:5]}_{timestamp}.jpg")
            return results
        except ImportError as e: 
            print(e)
            print(sys.exc_type)
            return []
        
    def predict_maskrcnn(self,patch):
        try: 
            results = self.model.detect([patch], verbose = 1)
            return results
        except ImportError as e: 
            print(e) 

    def predict_onnx(self,patch) :
        import time
        time.sleep(90)
        try: 
            results = self.model.session_run(patch)
            [print(f"Output {i} : {result}") for i, result in enumerate(results)]
            return results
        except AttributeError as e:
            print(e)
            print(sys.exc_type)

    def predict(self, patch):
        predict_func = self.get_predict_func(self.model_name)
        return predict_func(patch)

    def clear_gpu_memory(self):
        import torch
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def concurrent_prediction(self,patches) -> list :
        init_time = time.time()
        predictions = list(self.executor.map(self.predict,patches))
        if predictions is None: 
            print("No predictions were successful")
        else: 
            print(f"Frame {self.frame_id}/1800 inference done at {(time.time() - init_time):.2f} seconds")
            self.frame_id += 1
        return predictions

    def non_max_suppressions(self,predictions, iou_thres=0.5):
        if self.model_name == "Yolo" or self.model_name == "YOLO":
            import torchvision.ops as ops
            nms_detections = ops.nms(predictions[:,:4], predictions[:,4], iou_threshold=iou_thres)            
            return predictions[nms_detections] 

    # detections = sp.Detections.from_ultralytics(predictions)
    # detections = sp.Detections.from_yolov5(yolov5_results=predictions)
    # detections = detections.with_nms(threshold=0.5)

    def soft_nms_detections(self,predictions,method=2): 
        from keras import backend as K
        import tensorflow as tf
        array = np.asarray(predictions)
        with tf.compat.v1.Session() as sess:
            index = py_cpu_softnms(array[:,:4], array[:,4], method=method)
            selected_predictions = sess.run(tf.gather(array, index))
        
        return selected_predictions

    def gather_preds(self,patches,positions,predictions,frame_shape,stride,conf_thrs=0.4): 
        boxes = []
        scores = []
        class_ids =[]

        for (patch, pos, pred) in zip(patches,positions, predictions): 
            patch_height, patch_width,_ = patch.shape
            x_offset, y_offset = pos 
            left = x_offset * stride
            top = y_offset * stride

            if left + patch_width > frame_shape[1]:
                left = frame_shape[1] - patch_width
            if top + patch_height > frame_shape[0]:
                top = frame_shape[0] - patch_height

            prediction = pred.pandas().xyxy[0]
        
            for _,row in prediction.iterrows(): 
            
                if conf_thrs > row['confidence']:
                    continue
                
                boxes.append([row['xmin'] + left, row['ymin'] + top,
                             row['xmax'] + left, row['ymax'] + top] )
                
                scores.append(row['confidence'])
                class_ids.append(int(row['class']))   

        import torch
        boxes = torch.tensor(boxes, dtype=torch.float32) #If float16 it creates the nms_kernel not implemented for 'Half'
        scores = torch.tensor(scores, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int8)
        detections_tensor = torch.cat((boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)), dim=1)
        
        return detections_tensor
    
    def unique_elements(self,detections): 
        unique = set() 
        for det in detections: 
            box = det[0:4]
            unique.add(box)

        if len(unique) != len(detections): 
            print("Duplicates found")
            return [] 
        return detections 
        
    def tensor_to_det(self,tensor_predictions): 
        detections = []
        for det in tensor_predictions: 
            detections.append((det[0].item(),
                           det[1].item(),
                           det[2].item(),
                           det[3].item(), 
                           det[4].item(), 
                           int(det[5].item())))

        detections = self.unique_elements(detections)

        return detections 

    def draw_yolo_boxes(self,frame,patches,positions,predictions, full_image_shape, stride)->list:
        predictions = self.gather_preds(patches,positions,predictions,full_image_shape, stride)
        # predictions = self.non_max_suppressions(predictions)
        # detections = self.tensor_to_det(predictions)
        # detections = self.soft_nms_detections(detections,method=1)
        detections = predictions    
        for det in detections: 
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])
            score = det[4] 
            class_id = int(det[5])
            cv2.rectangle(frame,(x1,y1), (x2,y2),(0, 255, 0), 2)
            cv2.putText(frame, f'{self.class_names[class_id]} {score:.2f}', (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return detections

    def draw_maskrcnn_boxes(self,frame,patches,positions,predictions)->list:
        pass

    def draw_boxes(self, frame,  patches, positions, predictions,  full_image_shape, stride):
        draw_boxes_func = self.get_draw_func(self.model_name)
        return draw_boxes_func(frame, patches, positions, predictions, full_image_shape, stride)

