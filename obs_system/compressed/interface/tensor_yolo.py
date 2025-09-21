from obs_system.compressed.interface.utils import *
from obs_system.utils.logger import get_logger

import tensorrt_rtx as trt 
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
import pdb
import torch
import cv2
import os
import numpy as np
import time


logger = get_logger("obs_system."+__name__)


class TensorRTYOLO:

    def __init__(self, engine_path, conf_thres=0.5, iou_thres=0.5, fp16=True):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height=640 
        self.input_width=640
        # Runtime Phase: As per https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/python-api-docs.html#create-network-python
        self.logger_trt = trt.Logger(trt.Logger.WARNING)

        # Load TensorRT engine
        logger.debug(f"Loading TensorRT engine from {engine_path} ...")
        
        if engine_path.endswith('.onnx'): 
            builder = trt.Builder(self.logger_trt)

            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)

            # Populate the network definition from the ONNX representation with this parser. 
            parser = trt.OnnxParser(network, self.logger_trt)

            # Read the model file and process any errors 
            with open(engine_path, "rb") as f: 
                if not parser.parse(f.read()): 
                    for i in range(parser.num_errors): 
                        logger.debug(parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX model")

            # Build the TensorRT-RTX Engine
            config = builder.create_builder_config() 
            # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

            input_name = network.get_input(0).name
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, 
                (1,3,640, 640), 
                (16,3,640,640), 
                (32,3,640, 640)
            )

            config.add_optimization_profile(profile) 

            serialized_engine = builder.build_serialized_network(network, config)    

            if serialized_engine is None: 
                raise RuntimeError("Failed to build the TensorRT-RTX engine")

            save_path = os.getcwd() + "/yolo_trt.engine"
            try: 
                with open(save_path, "wb") as f: 
                    # Save the Engine for future use 
                    f.write(serialized_engine)
                logger.debug(f"Serialized engine saved to : {save_path}")
            except Exception as e: 
                logger.exception(f"could not save engine to disk: {e}")


            # Deserializing a TensorRT-RTX Engine - Read the engine file into a memory buffer. 
            runtime = trt.Runtime(self.logger_trt)

            # Diserialzie the engine from a serialized engine object
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        else: 
            if not engine_path.endswith(".engine"): 
                raise ValueError("Invalid engine file (must end with .engine)")

            # Deserializing a TensorRT-RTX Engine - Read the engine file into a memory buffer. 
            with open(engine_path, "rb") as f, trt.Runtime(self.logger_trt) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None: 
            raise RuntimeError(f"Could not deserialize engine")

        # Performing Inference based on IExecutionContext interface. 
        self.context = self.engine.create_execution_context()
        if self.context is None: 
            raise RuntimeError(f"Failed to create execution context")

        # Binding buffers for input and output, we don't use set_tensor_address(name, ptr) 
        for i in range(self.engine.num_io_tensors): 
            name = self.engine.get_tensor_name(i) 
            tshape = self.engine.get_tensor_shape(name)
            tdtype = self.engine.get_tensor_dtype(name)
            tmode = self.engine.get_tensor_mode(name)
            logger.debug(f"Tensor {i}: name={name}, shape={tshape}, dtype={tdtype}, mode={tmode}")

        self.input_name = None 
        self.output_names = []
        self.input_shape = self.engine.get_tensor_shape(self.input_name)

        for i in range(self.engine.num_io_tensors): 
            name = self.engine.get_tensor_name(i) 
            mode = self.engine.get_tensor_mode(name) 

            if mode == trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = name 

            if mode == trt.TensorIOMode.OUTPUT: 
                self.output_names.append(name)

        if self.input_name is None: 
            raise RuntimeError("Could not determine input tensor shape")

        self.runtime_input_shape = (32,3,640,640)
        self.context.set_input_shape(self.input_name, self.runtime_input_shape)

        # Initialize the CUDA Stream
        self.stream = cuda.Stream()

        in_dtype = self.engine.get_tensor_dtype(self.input_name)
        in_nptype = trt.nptype(in_dtype)
        in_size = int(trt.volume(self.runtime_input_shape)) * np.dtype(in_nptype).itemsize 
        self.d_input = cuda.mem_alloc(in_size)
        
        # Register the input pointer to the context which is required from execute_async_v3 
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        
        self.d_outputs = []
        self.host_outputs = []       

        for name in self.output_names: 

            shape = tuple(self.context.get_tensor_shape(name))
            odtype = self.engine.get_tensor_dtype(name) 
            ontype = trt.nptype(odtype)

            h_out = np.empty(shape, dtype=ontype)
            d_out = cuda.mem_alloc(h_out.nbytes) 

            self.d_outputs.append(d_out) 
            self.host_outputs.append(h_out) 
            self.context.set_tensor_address(name, int(d_out))
            

        logger.debug("TensorRTYOLO initialized successfully.")


    def __call__(self, image):
        return self.detect_objects(image)


    def prepare_input_image(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image 
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 and 1 
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32)
        return input_tensor


    def inference(self, input_tensor, return_numpy=False):
        """
        input_tensor: torch.Tensor [B,3,H,W] or np.ndarray
        """
        start = time.perf_counter()
        assert input_tensor.is_cuda and input_tensor.dtype==torch.float32 and input_tensor.is_contiguous() 

        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy().astype(np.float32)
        shape = tuple(input_tensor.shape)
        
        if tuple(self.context.get_tensor_shape(self.input_name))!= shape: 
            self.context.set_input_shape(self.input_name, shape)


            in_bytes = int(trt.volume(shape)) * np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name))).itemsize
            if not hasattr(self, "d_input") or getattr(self.d_input, "size", 0) < in_bytes: 
                if hasattr(self, "d_input"): self.d_input.free() 
                self.d_input = cuda.mem_alloc(in_bytes) 

            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.host_outputs, self.d_outputs = [], [] 

            for name in self.output_names:
                oshape = tuple(self.context.get_tensor_shape(name))
                odtype = trt.nptype(self.engine.get_tensor_dtype(name))
                h_out = np.empty(oshape, dtype=odtype) 
                d_out = cuda.mem_alloc(h_out.nbytes) 
                self.host_outputs.append(h_out)
                self.d_outputs.append(d_out) 
                self.context.set_tensor_address(name, int(d_out))
            




        # Copy input to GPUa, common to enqueue asynchronous transferd before and after the kernels to move data to the GPU
        cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)

        # Run inference
        self.context.execute_async_v3( stream_handle=self.stream.handle)

        # Copy outputs back
        for i,(h_out,d_out) in enumerate(zip(self.host_outputs, self.d_outputs)):
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)

        # Determine when inference (and asynchronous transfers) are complete
        self.stream.synchronize()

        outputs = [torch.from_numpy(out) for out in self.host_outputs]

        logger.debug(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

        if return_numpy:
            return [out.numpy() for out in outputs]

        return outputs


    def detect_objects(self, im):
        if isinstance(im, torch.Tensor):

            self.img_height, self.img_width = im.shape[2], im.shape[3]
            outputs = self.inference(im, return_numpy=False)
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        elif isinstance(im, np.ndarray):
            self.img_height, self.img_width = im.shape[:2]
            input_tensor = self.prepare_input_image(im)
            outputs = self.inference(input_tensor, return_numpy=False)
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids


    def warmup(self, micro=32, img_size=(640, 640), device="cuda:0", warmup_sessions=5):
        dummy_batch = torch.zeros(
            (micro, 3, img_size[0], img_size[1]),
            device=device,
            dtype=torch.float32
        )

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

            if pred.min() < 0 or pred.max() > 1:
                cls = 1/(1 + torch.exp(-cls))
                if isinstance(obj, torch.Tensor): 
                    obj = 1/(1 + torch.exp(-obj))

            for preds in batch_images:
                predictions = preds.T
                scores = predictions[:, 4:].max(dim=-1).values * (
                    obj if torch.is_tensor(obj) else 1.0
                )
                mask = scores > self.conf_threshold

                if not torch.any(mask):
                    all_boxes.append(torch.empty((0, 4), dtype=torch.float16))
                    all_scores.append(torch.empty((0,), dtype=torch.float16))
                    all_class_ids.append(torch.empty((0,), dtype=torch.int32))
                    continue

                predictions = predictions[scores > self.conf_threshold, :]
                scores = scores[scores > self.conf_threshold]
                class_ids = torch.argmax(predictions[:, 4:], dim=1)
                boxes = self.extract_boxes(predictions)

                indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
                if indices is not None and len(indices) > 0:
                    all_boxes.append(boxes[indices])
                    all_scores.append(scores[indices])
                    all_class_ids.append(class_ids[indices])
                else:
                    all_boxes.append(torch.empty((0, 4), dtype=torch.float16))
                    all_scores.append(torch.empty((0,), dtype=torch.float16))
                    all_class_ids.append(torch.empty((0,), dtype=torch.int32))
        
        return all_boxes, all_scores, all_class_ids


    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        if not (self.input_width == 640 and self.input_height == 640):
            boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes


    def rescale_boxes(self, boxes):
        if isinstance(boxes, torch.Tensor):
            input_shape = torch.tensor(
                [self.input_width, self.input_height, self.input_width, self.input_height]
            )
            boxes = torch.div(boxes, input_shape)
            boxes *= torch.tensor(
                [self.img_width, self.img_height, self.img_width, self.img_height]
            )
        else:
            input_shape = np.array(
                [self.input_width, self.input_height, self.input_width, self.input_height]
            )
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array(
                [self.img_width, self.img_height, self.img_width, self.img_height]
            )
        return boxes

   
