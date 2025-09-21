from obs_system.compressed.interface.utils import *
from obs_system.utils.logger import get_logger

import torch
import cv2 
import time 
import onnxruntime as ort 
import numpy as np

logger = get_logger("obs_system."+__name__)

# This was inspired by : https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/tree/library

class CompressedYOLO:

    def __init__(self, path, conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)


    def __call__(self, image):
        return self.detect_objects(image)


    def initialize_model(self, path):
        logger.debug(f"Available providers for ORT: {ort.get_available_providers()}") 
        self.session = ort.InferenceSession(
                path, 
                providers=[
                    # ("TensorrtExecutionProvider", {
                    #     "trt_fp16_enable":True, 
                    #     "trt_engine_cache_enable":True, 
                    #     "trt_engine_cache_path":"./trt_cache", 
                    #     "trt_timing_cache_enable": True, 
                    #     "trt_max_workspace_size":1<<30
                    # }),
                    "CUDAExecutionProvider", "CPUExecutionProvider"]
                )

        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, im):
        if isinstance(im, torch.Tensor): 
            # Perform inference on the tiles
            self.img_height, self.img_width = im[0].shape[1], im[0].shape[2]
            color_convert = False 

            def check_tensor(im): 
                if not color_convert: 
                    im = im.reshape(im.shape[1], im.shape[2], im.shape[0]) 
                    im = im.cpu().numpy() 
                    check = np.argmax(im.mean((0,1))) 
                
                    if check == 2: 
                        return True 
                    return False 
            
            BRG_to_RGB = check_tensor(im[0])

            if BRG_to_RGB: im = im.flip(1)

            input_tensor = im #this is a tensor with 16 images. 
            outputs = self.inference(input_tensor)
            
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)
            
        elif isinstance(im, np.ndarray): 
            self.img_height, self.img_width = im.shape[:2]
            input_tensor = self.prepare_input_image(im)
            input_tensor = input_tensor.detach().cpu().numpy().astype(np.float32) 

        return self.boxes, self.scores, self.class_ids
    

    def prepare_input_image(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (int(self.input_width), int(self.input_height)))

        # Scale input pixel values to 0 and 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor, return_numpy=False):

        """ 
        This class method at its current implementation takes: 
            input_tensor: torch.Tensor | List(np.ndarray)

        Uses the IObinding capabilities from ORT to inference the tiles of the image 
        efficiently through binding the same device for input and output. 

        """
        start = time.perf_counter()

        assert input_tensor.is_cuda and input_tensor.dtype==torch.float32 and input_tensor.is_contiguous() 

        device, device_id = input_tensor.device.type, input_tensor.device.index
        io = self.session.io_binding() 
        shape = tuple(input_tensor.shape) 
        ptr = int(input_tensor.data_ptr()) 

        io.bind_input(name=self.input_names, 
            device_type=device, device_id=device_id, 
            element_type=np.float32, shape=shape, 
            buffer_ptr=ptr
        )

        for out in self.session.get_outputs(): 
            io.bind_output(
                    name=out.name, 
                    device_type=device, 
                    device_id=device_id, 
                    element_type=np.float32
            ) 

        self.session.run_with_iobinding(io) 

        if return_numpy: 
            outputs = io.get_outputs() 
            outputs = [o.numpy() for o in outputs]

        else: 
            outputs = io.get_outputs() 
            # outputs = None
            outputs = [torch.from_numpy(o.numpy()) for o in outputs] 

        logger.debug(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs


    def warmup(self, micro:int=32, img_size=(640,640), device='cuda:0', warmup_sessions:int=5): 
        device = torch.device(device)
        dummy_batch = torch.zeros((micro, 3, img_size[0], img_size[1]), device=device, dtype=torch.float32)

        for _ in range(warmup_sessions): 
            self.inference(dummy_batch, return_numpy=False)
            

    def process_output(self, output):
        batch_images = output[0] 

        all_boxes, all_scores, all_class_ids = [], [], [] 
        if isinstance(batch_images, torch.Tensor): 
            pred = torch.transpose(batch_images, 1, 2)
            
            if pred.shape[-1] == 85: 
                obj = pred[..., 4:5] 
                cls = pred[..., 5:] 
            else: 
                obj = 1
                cls = pred[..., 4:]

            for preds in batch_images: 
                predictions = preds.T 
                scores = predictions[:,4:].max(dim=-1).values * (obj if np.isscalar(obj)==False else 1.0)
                mask = scores > self.conf_threshold 

                if not torch.any(mask): 
                    all_boxes.append(torch.empty((0,4), dtype=torch.float16))
                    all_scores.append(torch.empty((0,), dtype=torch.float16)) 
                    all_class_ids.append(torch.empty((0,), dtype=torch.int32))
                    continue
       
                predictions = predictions[scores>self.conf_threshold, :] 
                scores = scores[scores > self.conf_threshold] 
                class_ids = torch.argmax(predictions[:,4:], axis=1)
                boxes = self.extract_boxes(predictions)

                indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold) 
                if indices: 
                    all_boxes.append(boxes[indices]) 
                    all_scores.append(scores[indices]) 
                    all_class_ids.append(class_ids[indices]) 

                else: 
                    all_boxes.append(torch.empty((0,4), dtype=torch.float16))
                    all_scores.append(np.empty((0,), dtype=np.float16))
                    all_class_ids.append(np.empty((0,), dtype=np.int32))
        
        else: 

            pred = np.transpose(batch_images, (0,2,1)) 
            if pred.shape[-1] == 85: 
                obj = pred[..., 4:5] 
                cls = pred[..., 5:] 
            else: 
                obj = 1 
                cls = pred[..., 4:] 

            if pred.min() < 0 or pred.max() > 1: 
                cls = 1/(1+np.exp(-cls)) 
                if isinstance(obj, np.ndarray): 
                    obj = 1/(1 + np.exp(-obj)) 

        for preds in batch_images: 

            predictions = preds.T
            
            # Filter predictions 
            scores = np.max(predictions[:,4:], axis=-1) * (obj if np.isscalar(obj)==False else 1.0) 
            mask = scores > self.conf_threshold 
            
            if not np.any(mask): 
                all_boxes.append(np.empty((0,4), dtype=np.float32))
                all_scores.append(np.empty((0,), dtype=np.float32))
                all_class_ids.append(np.empty((0,), dtype=np.int64)) 
                continue 

            predictions = predictions[scores>self.conf_threshold, :]
            scores = scores[scores>self.conf_threshold]

            # Get class with highest confidence 
            class_ids = np.argmax(predictions[:,4:], axis=1)
            boxes = self.extract_boxes(predictions) 

            # Non-Maima suppression for overlapping bounding boxes
            indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold) 
            
            if indices: 
                all_boxes.append(boxes[indices]) 
                all_scores.append(scores[indices]) 
                all_class_ids.append(class_ids[indices]) 

            else: 
                all_boxes.append(np.empty((0,4), dtype=np.float32))
                all_scores.append(np.empty((0,), dtype=np.float32))
                all_class_ids.append(np.empty((0,), dtype=np.float32))
        
        logger.info("Post Process completed...")
        return all_boxes, all_scores, all_class_ids 


    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        if not (self.input_width == 640 and self.input_height == 640): 
            boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes


    def rescale_boxes(self, boxes):

        if isinstance(boxes, torch.Tensor): 
            input_shape = torch.tensor([self.input_width, self.input_height, self.input_width, self.input_height])
            boxes = torch.div(boxes, input_shape)
            boxes *= torch.tensor([self.img_width, self.img_height, self.img_width, self.img_height])
        # Rescale boxes to original image dimensions
        else: 
            input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        #self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_names = model_inputs[0].name  
        self.input_shape = model_inputs[0].shape

        # Check for dynamic shape
        if isinstance(self.input_shape[2], str) or isinstance(self.input_shape[3], str):

            self.input_height = 640
            self.input_width = 640
        else:
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])


    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
