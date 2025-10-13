from obs_system.compressed.interface.utils import *
from obs_system.utils.logger import get_logger

import pdb
import tensorrt as trt 
import gc
import torch
import cv2
import os
import numpy as np
import time


logger = get_logger("obs_system."+__name__)


class TensorRTYOLO:

    def __init__(self, engine_path, conf_thres=0.5, iou_thres=0.5, fp16=False, int8=False, strip_weights=False):
        
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height=640 
        self.input_width=640

        self.__input_name = None 
        self.__shape_input_model = [32, 3, 640, 640]
        self.__output_names = [] 

        self.__device = torch.device("cuda", torch.cuda.current_device()) 
        torch.cuda.set_device(self.__device)
        self.__stream = torch.cuda.Stream(device=self.__device)
        self.__in_dev = None 
        self.__out_devs = {}

        self.__fp16 = fp16 
        self.__int8 = int8
        self.__strip_weights = strip_weights
        self.__TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Runtime Phase: As per https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/python-api-docs.html#create-network-python

        # Load TensorRT engine
        logger.debug(f"Loading TensorRT engine from {engine_path} ...")
        
        save_path = os.getcwd() + f"/obs_system/compressed/yolo_trt_{'fp16' if self.__fp16 else 'nofp16'}_{'int8' if self.__int8 else 'noint8'}.engine"
        if not os.path.exists(save_path): 
            self.__build_engine__(engine_path, save_path)
            self.__load_engine__(load_path=save_path)
            # self.__set_stream(runtime_input_shape=(32,3, 640, 640))
        else: 
            self.__load_engine__(load_path=save_path)
            # self.__set_stream(runtime_input_shape=(32,3,640,640))
        

    def __call__(self, image, debug=False):
        return self.detect_objects(image, debug=debug)


    def __stream_handler(self): 
        return torch.cuda.current_stream(self.__device).cuda_stream


    def __build_engine__(self, engine_path, save_path:str): 
        
            # TensorRT Optimizer
            builder = trt.Builder(self.__TRT_LOGGER) 

            # Config for the builder and network 
            config = builder.create_builder_config() 

            # Set Cache 
            cache = config.create_timing_cache(b"") 
            config.set_timing_cache(cache, ignore_mismatch=False)
            
            #Define max Workspace for memory Limit 
            max_workspace = (1 < 30) 
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace)

            # Create the Network Definition (representation of a model in TensorRT) 
            net_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(net_flag) 

            # Populate the network definition from the ONNX representation
            parser = trt.OnnxParser(network, self.__TRT_LOGGER) 

            # Parse the ONNX model 
            with open(engine_path, "rb") as rf: 
                    if not parser.parse(rf.read()): 
                            logger.error(f"Failed to parse the ONNX model file {engine_path}")
                            for i in range(parser.num_errors): 
                                    logger.debug(parser.get_error(i))
                            raise RuntimeError("Failed to parse ONNX Model")

            inputs = [network.get_input(i) for i in range(network.num_inputs)] 
            outputs = [network.get_output(i) for i in range(network.num_outputs)]

            for input in inputs: 
                    logger.debug(f"Model {input.name} shape: {input.shape} {input.dtype}")
            for output in outputs:
                    logger.debug(f"Model {output.name} shape: {output.shape} {output.dtype}") 

            profile = builder.create_optimization_profile() 
            min_shape = [1] + self.__shape_input_model[-3:] 
            opt_shape = [int(self.__shape_input_model[0]/2)] + self.__shape_input_model[-3:] 
            max_shape = self.__shape_input_model

            for input in inputs: 
                profile.set_shape(
                    input.name, 
                    min_shape, opt_shape, max_shape
                )

            config.add_optimization_profile(profile)

            # Default is TF32 
            if self.__fp16: 
                config.set_flag(trt.BuilderFlag.FP16) 
            elif self.__int8: 
                config.set_flag(trt.BuilderFlag.INT8) 

            if self.__strip_weights: 
                config.set_flag(trt.BuilderFlag.STRIP_PLAN) 
            else: 
                config.flags &= ~(1 << int(trt.BuilderFlag.STRIP_PLAN))

            engine_bytes = builder.build_serialized_network(network, config) 

            if engine_bytes is None: 
                raise RuntimeError("Could not deserialize engine")

            with open(save_path, "wb") as sf: 
                sf.write(engine_bytes)

            


    def __load_engine__(self, load_path:str="", serialized_engine=None): 

        runtime = trt.Runtime(self.__TRT_LOGGER) 

        # Deserialization on a TensorRT (or TensorRTX) engine - Read engine file into memory buffer. 
        if not os.path.exists(load_path) and serialized_engine is not None: 
           self.__engine = runtime.deserialize_cuda_engine(serialized_engine) 

        elif os.path.exists(load_path) and serialized_engine is None: 
            with open(load_path, "rb") as f: 
                self.__engine = runtime.deserialize_cuda_engine(f.read())
            logger.info("Engine Create through the save file")

        else: 
            raise ValueError(f"There is no engine given and no serialized engine is provided")

        if self.__engine is None: 
            raise RuntimeError("Could not deserialize engine")
        
        # Performing Inference based on IExecutionContext interface
        self.__context = self.__engine.create_execution_context() 

        if self.__context is None: 
            raise RuntimeError("Failed to creatre execution context")

        # Binding buffers for input/output, we don't use set_tensor_address(name, ptr)
        for i in range(self.__engine.num_io_tensors): 
            name = self.__engine.get_tensor_name(i)
            tshape = self.__engine.get_tensor_shape(name) 
            tdtype = self.__engine.get_tensor_dtype(name) 
            tmode = self.__engine.get_tensor_mode(name) 
            logger.debug(f"Tensor {i}: name={name}, shape={tshape}, dtype={tdtype}, mode={tmode}")

        self.__input_name = None
        self.__output_names = [] 

        for i in range(self.__engine.num_io_tensors):
            name = self.__engine.get_tensor_name(i) 
            mode = self.__engine.get_tensor_mode(name) 

            if mode == trt.TensorIOMode.INPUT and self.__input_name is None: 
                self.__input_name = name 

            if mode == trt.TensorIOMode.OUTPUT: 
                self.__output_names.append(name) 
            
        if self.__input_name is None: 
            raise RuntimeError("Could not determine input tensor shape")
 
        # self.__context.set_input_shape(self.__input_name,self.__shape_input_model) 


    def __trt_to_tensor_dtype(self, trt_dtype: trt.DataType)->torch.dtype: 
        return {
                trt.DataType.FLOAT : torch.float32, 
                trt.DataType.HALF : torch.float16, 
                trt.DataType.INT32 : torch.int32, 
                trt.DataType.INT8 : torch.int8, 
                trt.DataType.BOOL : torch.bool, 
        }[trt_dtype]
            



    def getter_name(self): 
        return {
                "engine" : self.__engine, 
                "input_name": self.__input_name, 
                "output_names":self.__output_names, 
                "context":self.__context
        }

    
    def __set_stream(self, runtime_input_shape=(32, 3, 640,640)): 
            
            # initialize Cuda Stream 
            # self.__stream = cuda.Stream() 
            # Stream is now initiallized with torch.
    
            try: 
                self.__context.set_optimnization_profile_async(0, self.__stream.cuda_stream) 
            except Exception as e: 
                pass 

            self.__context.set_input_shape(self.__input_name, runtime_input_shape) 

            in_trt_dtype = self.__engine.get_tensor_dtype(self.__input_name) 
            in_torch_dtype = self.__trt_to_tensor_dtype(in_trt_dtype)

            if (self.__in_dev is None or tuple(self.__in_dev.shape) != runtime_input_shape or self.__in_dev.dtype != in_torch_dtype): 
                self.__in_dev = torch.empty(
                    runtime_input_shape, device="cuda", dtype=in_torch_dtype, memory_format=torch.contiguous_format
                )

            # in_dtype = self.__engine.get_tensor_dtype(self.__input_name)
            # in_nptype = trt.nptype(in_dtype) 
            # in_size = int(trt.volume(runtime_input_shape)) * np.dtype(in_nptype).itemsize
            # self.__d_input = cuda.mem_alloc(in_size) 
        
            # Register the input pointer to the context which is required from execute_async_v3 
            self.__context.set_tensor_address(self.__input_name, int(self.__in_dev.data_ptr()))

            # self.__d_outputs = [] 
            # self.__host_outputs = [] 
            
            self.__out_devs.clear() 
            for name in self.__output_names: 
                    shape = tuple(self.__context.get_tensor_shape(name)) 
                    odtype = self.__engine.get_tensor_dtype(name) 
                    ontype = self.__trt_to_tensor_dtype(odtype)  

                    out_dev = torch.empty(shape, device=self.__device, dtype=ontype)
                    self.__out_devs[name] = out_dev
                    self.__context.set_tensor_address(name, int(out_dev.data_ptr()))

            logger.debug("TensorRTYOLO Initialized correctly")


    def prepare_input_image(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image 
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 and 1 
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float16)
        return input_tensor


    def inference(self, input_tensor, return_numpy=False, verbose=False):
        """
        input_tensor: torch.Tensor [B,3,H,W] or np.ndarray
        """
        start = time.perf_counter()
        
        # assert input_tensor.is_cuda and input_tensor.is_contiguous() 
        
        if isinstance(input_tensor, np.ndarray): 
            torch_in = torch.from_numpy(input_tensor) 
        elif isinstance(input_tensor, torch.Tensor): 
            torch_in = input_tensor 
        else : 
            raise ValueError("input_tensor must be np.ndarray or torch.Tensor")

    
        in_trt_dtype = self.__engine.get_tensor_dtype(self.__input_name) 
        in_torch_dtype = self.__trt_to_tensor_dtype(in_trt_dtype) 
        if torch_in.dtype != in_torch_dtype: 
            torch_in = torch_in.to(dtype=in_torch_dtype) 

        shape = tuple(int(x) for x in torch_in.shape) 
        
        if (self.__in_dev is None) or (tuple(self.__in_dev.shape) != shape): 
            self.__set_stream(shape) 

        if torch_in.device != self.__device: 
            torch_in = torch_in.to(self.__device, non_blocking=True)
        torch_in = torch_in.contiguous()

        with torch.cuda.stream(self.__stream): 
            self.__in_dev.copy_(torch_in, non_blocking=True) 
            self.__context.execute_async_v3(stream_handle=self.__stream_handler())


        # logger.debug(f"exec on device {self.__device}, stream {int(self.__stream.cuda_stream)}")
        # for n,t in self.__out_devs.items():
        #     logger.debug(f"{n}: dev={t.device}, ptr={t.data_ptr()}")
        # logger.debug(f"in_dev: dev={self.__in_dev.device}, ptr={self.__in_dev.data_ptr()}")
        #

        outputs_gpu = [self.__out_devs[name] for name in self.__output_names] 
        if return_numpy: 
            with torch.cuda.stream(self.__stream): 
                outs_cpu = [t.detach().cpu() for t in outputs_gpu] 

            self.__stream.synchronize() 
            outs_np = [t.numpy() for t in outs_cpu] 
            if verbose: 
                logger.debug(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

            return outs_np

        else: 
            self.__stream.synchronize() 
            if verbose: 
                logger.debug(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
            return outputs_gpu


        #     if input_tensor.dtype == torch.float32: 
        #         input_tensor = input_tensor.to(dtype=torch.float16) 
        #     input_tensor = input_tensor.detach().cpu().numpy()
        # else: 
        #     if input_tensor.dtype not in (np.float16, np.float32) : 
        #         input_tensor = input_tensor.astype(np.float16, copy=False) 
        #     input_tensor = np.ascontiguousarray(input_tensor)

        # shape = tuple(input_tensor.shape)
        
        # if tuple(self.__context.get_tensor_shape(self.__input_name))!= shape: 
        #     self.__context.set_input_shape(self.__input_name, shape)
        #
        #     in_bytes = int(trt.volume(shape)) * np.dtype(trt.nptype(self.__engine.get_tensor_dtype(self.__input_name))).itemsize
        #     if not hasattr(self, "d_input") or getattr(self.__d_input, "size", 0) < in_bytes: 
        #         if hasattr(self, "d_input"): self.__d_input.free() 
        #         self.__d_input = cuda.mem_alloc(in_bytes) 
        #
        #     self.__context.set_tensor_address(self.__input_name, int(self.__d_input))
        #     self.__host_outputs, self.__d_outputs = [], [] 
        #
        #     for name in self.__output_names:
        #         oshape = tuple(self.__context.get_tensor_shape(name))
        #         odtype = trt.nptype(self.__engine.get_tensor_dtype(name))
        #         h_out = np.empty(oshape, dtype=odtype) 
        #         d_out = cuda.mem_alloc(h_out.nbytes) 
        #         self.__host_outputs.append(h_out)
        #         self.__d_outputs.append(d_out) 
        #         self.__context.set_tensor_address(name, int(d_out))
            
        # # Copy input to GPU, common to enqueue asynchronous transfered before and after the kernels to move data to the GPU
        # cuda.memcpy_htod_async(self.__d_input, input_tensor, self.__stream)
        # 
        # # Run inference
        # self.__context.execute_async_v3( stream_handle=self.__stream.handle)
        #
        # # Copy outputs back
        # for (h_out,d_out) in (zip(self.__host_outputs, self.__d_outputs)):
        #     cuda.memcpy_dtoh_async(h_out, d_out, self.__stream)
        #
        # # Determine when inference (and asynchronous transfers) are complete
        # self.__stream.synchronize()
        #
        # outputs = [torch.from_numpy(out) for out in self.__host_outputs]
        #
        # if verbose:
        #     logger.debug(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        #
        # if return_numpy:
        #     return [out.numpy() for out in outputs]

        return outputs


    def detect_objects(self, im, debug=False):
        if isinstance(im, torch.Tensor):

            self.img_height, self.img_width = im.shape[2], im.shape[3]
            color_convert = False  

            def check_tensor(im): 
                if not color_convert:
                    im = im.reshape(im.shape[1], im.shape[2], im.shape[0]) 
                    im = im.cpu().numpy() 
                    check = np.argmax(im.mean((0,1)))
                    
                    if check == 2: 
                        return True 
                    return False

            BGR_to_RGB = check_tensor(im[0]) 
            if BGR_to_RGB: 
                im = im.flip(1)

            outputs = self.inference(im, return_numpy=False, verbose=debug)
            self.boxes, self.scores, self.class_ids = self.process_output(outputs, debug=debug)

        elif isinstance(im, np.ndarray):
            self.img_height, self.img_width = im.shape[:2]
            input_tensor = self.prepare_input_image(im)
            outputs = self.inference(input_tensor, return_numpy=False, verbose=debug)
            self.boxes, self.scores, self.class_ids = self.process_output(outputs, debug=debug)

        return self.boxes, self.scores, self.class_ids


    def warmup(self, micro=32, img_size=(640, 640),dtype=torch.float16, device="cuda:0", warmup_sessions=5):
        
        dummy_batch = torch.zeros(
            (micro, 3, img_size[0], img_size[1]),
            device=self.__device,
            dtype=dtype
        )

        for _ in range(warmup_sessions):
            self.inference(dummy_batch, return_numpy=False)


    def process_output(self, output, debug=False):
        
        outputs = [o if isinstance(o, torch.Tensor) else torch.from_numpy(o) for o in output] 
        
        all_boxes, all_scores, all_class_ids = [], [], []

        if len(outputs) == 1: 

            #Raw Head
            batch_images = outputs[0] 
            pred = torch.transpose(batch_images, 1,2)  
            if pred.dim() != 3: 
                if debug: 
                    logger.debug(f"Unexpected raw head dim: {pred.shape}") 
                return [torch.empty((0,4))], [torch.empty(0)], [torch.empty(0, dtype=torch.int32)]

            if pred.shape[1] in (84,85) and pred.shape[2] not in (84,85):
                preds = pred.permute(0,2,1).contiguous()
                CLS = preds.shape[1] 
            elif pred.shape[2] in (84,85): 
                CLS = pred.shape[2] 
            else: 
                if debug: logger.debug("Cannot Identify channel axis") 
                return  [torch.empty((0,4))], [torch.empty(0)], [torch.empty(0, dtype=torch.int32)]

            if CLS == 85: 
                objs = pred[..., 4:5] 
                cls_logits = pred[..., 5:] 
                # obj = torch.sigmoid(objs_logits)
            else: 
                obj = 1
                cls_logits = pred[..., 4:]

            cls_probs = torch.sigmoid(cls_logits)               
            # scores, class_ids = cls_probs.max(dim=-1) 
            # pdb.set_trace()
            # scores = preds[:,4:].max(dim=-1).values * (objs_probs if np.isscalar(objs_probs)==False else 1.0) 
            # conf_mask = scores > self.conf_threshold 
            del pred
            gc.collect()
            for b, predictions in enumerate(batch_images): 

                predictions = predictions.T
                scores = predictions[:,4:].max(dim=-1).values * (obj if np.isscalar(obj)==False else 1.0)

                # best_cls, cls_ids = cls_probs[b].max(dim=-1)
                # scores = best_cls * (obj.squeeze(-1) if obj is not None else 1.0) 
                conf_mask = scores > self.conf_threshold
                if not torch.any(conf_mask): 
                    all_boxes.append(torch.empty((0, 4), dtype=torch.float16))
                    all_scores.append(torch.empty((0,), dtype=torch.float16))
                    all_class_ids.append(torch.empty((0,), dtype=torch.int32))
                    continue 

                predictions = predictions[scores>self.conf_threshold, :] 
                sel_scores = scores[scores > self.conf_threshold] 
                sel_cls = torch.argmax(predictions[:,4:],dim=1)  
                boxes = self.extract_boxes(predictions) 
                
                keep = self.multi_class_nms(boxes, sel_scores, sel_cls, self.iou_threshold) 
                
                if keep is not None and len(keep) > 0:
                    all_boxes.append(boxes[keep])
                    all_scores.append(sel_scores[keep])
                    all_class_ids.append(sel_cls[keep])
                    
                else:
                    all_boxes.append(torch.empty((0, 4), dtype=torch.float16))
                    all_scores.append(torch.empty((0,), dtype=torch.float16))
                    all_class_ids.append(torch.empty((0,), dtype=torch.int32))

        elif len(outputs) == 4: 
            # ---- END2END (post-NMS) ----
            num_dets, det_boxes, det_scores, det_classes = outputs

            B = det_boxes.shape[0]
            for b in range(B):

                k = int(num_dets[b].item())

                if k <= 0:
                    all_boxes.append(torch.empty((0, 4)))
                    all_scores.append(torch.empty((0,)))
                    all_class_ids.append(torch.empty((0,), dtype=torch.int64))
                    continue

                boxes = det_boxes[b, :k, :]                         
                scores = det_scores[b, :k]
                cls_ids = det_classes[b, :k].to(torch.int64)

                mask = scores > self.conf_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                cls_ids = cls_ids[mask]

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(cls_ids)
        
        else: 
            if debug: logger.debug(f"Unexpected number of outputs: {len(outputs)}") 
            all_boxes.append(torch.empty((0,4)))
            all_scores.append(torch.empty((0,)))
            all_class_ids.append(torch.empty((0,), dtype=torch.int64))

                
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
                [self.input_width, self.input_height, self.input_width, self.input_height], 
                device=boxes.device, dtype=boxes.dtype
            )
            boxes = torch.div(boxes, input_shape)
            boxes *= torch.tensor(
                [self.img_width, self.img_height, self.img_width, self.img_height], 
                device=boxes.device, dtype=boxes.dtype
            )
        else:
            input_shape = np.array(
                [self.input_width, self.input_height, self.input_width, self.input_height]
            )
            boxes = np.divide(boxes, input_shape, dtype=np.float16)
            boxes *= np.array(
                [self.img_width, self.img_height, self.img_width, self.img_height]
            )
        return boxes

     
    def multi_class_nms(self, in_boxes, in_scores, in_class_ids, iou_threshold:float): 

        def _device_of(x): 
            return x.device if isinstance(x,torch.Tensor) else None 

        dev = _device_of(in_boxes) or _device_of(in_scores) or _device_of(in_class_ids) or self.__device=="cpu" 

        boxes = in_boxes if isinstance(in_boxes,torch.Tensor) else torch.as_tensor(in_boxes)
        scores = in_scores if isinstance(in_scores, torch.Tensor) else torch.as_tensor(in_scores) 
        class_ids = in_class_ids if isinstance(in_class_ids, torch.Tensor) else torch.as_tensor(in_class_ids) 

        boxes = boxes.to(device=dev, dtype=torch.float32).contiguous() 
        scores = scores.to(device=dev, dtype=torch.float32).contiguous().reshape(-1)
        class_ids = scores.to(device=dev, dtype=torch.int64).contiguous().reshape(-1)

        finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores) 
        if finite.sum().item() != boxes.size(0): 
            boxes, scores, class_ids = boxes[finite], scores[finites] , class_ids[finite] 

        if boxes.numel() == 0: 
            return torch.empty((0,), dtype=torch.long, device=dev)
        
        keep = [] 
        unique_classes = torch.unique(class_ids) 
        for cid in unique_classes: 
            mask = (class_ids == cid) 
            if not mask.any(): 
                continue 

            sel_boxes = boxes[mask] 
            sel_scores = scores[mask] 

            kept_local = nms(sel_boxes, sel_scores, iou_threshold) 
            if kept_local.numel() != 0: 
                global_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[kept_local] 
                keep.append(global_idx) 
        if keep: 
            return torch.cat(keep) 

        return torch.empty((0,), dtype=torch.long, device=dev)
        

        

