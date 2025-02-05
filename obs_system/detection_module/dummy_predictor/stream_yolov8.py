from obs_system.detection_module.interface.streaming import YOLOStreamer
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.build import load_inference_source
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.data.augment import classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics import YOLO 
from ultralytics.engine.results import Results
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path 
from collections import defaultdict
import torch 
import numpy as np
import cv2 
import re
import torchvision.ops as operation
import warnings 
import os
import time
from shapely.geometry.point import Point


class Yolov8Streamer(YOLOStreamer):

    # Ultralytics YOLO 🚀, AGPL-3.0 license

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg, overrides, _callbacks)
        self.speed = {} 
        self.track_history = None
        self.trackers = [] 
        

    def warmup(self, imgsz=(1, 3, 640, 640)): 
        """Pytorch uses graph optimizations that are triggered with the first inference."""

        if self.device != 'cpu': 
            im = torch.empty(*imgsz, dtype=torch.float, device=self.device)
            y = self.model(im, show=False)
            if isinstance(y, (list,tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else: 
                return self.from_numpy(y)
            

    def from_numpy(self, x): 
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    def __call__(self, source=None, model=None, stream=False, mqtt_broker=None, producer_flag=None, queue=None, *args, **kwargs):
        
        """Performs inference on an image, video or stream."""
        self.mqtt_interface = mqtt_broker
        self.args.stream_buffer = True
        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if  os.path.isfile(source)  else source,
                             model=model,
                             producer_flag=producer_flag,
                             queue=queue)
        except KeyboardInterrupt as ke: 
            warnings.warn(KeyboardInterrupt.__doc__)
            if producer_flag is not None: 
                producer_flag.value=False
            cv2.destroyAllWindows()
            return 


    def pre_transform(self, im): 
        return super().pre_transform(im)
    

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        
        not_tensor = not isinstance(im, torch.Tensor)
        self.orig_shape = im[0].shape if isinstance(im, list) else im.shape
        
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        
        if not isinstance(self.model, YOLO): 
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        else: 
            im = im.float()  # uint8 to fp32

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0
        return im


    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )

        if isinstance(self.model, YOLO): 
            return self.model.track(im, augment=self.args.augment, visualize=visualize,embed=self.args.embed, conf=0.4, iou=0.5, verbose=False, show=False, persist=True, tracker="bytetrack.yaml", stream_buffer=self.args.stream_buffer)
        else: 
            return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
    

    def postprocess(self, preds, img, orig_img): 
        return super().postprocess(preds, img, orig_img)
    
    
    def predict_cli(self, source, model, producer_flag=None, queue=None): 
        return super().predict_cli(source, model, producer_flag, queue) #sourcery skip: remove-empty-nested-block noqa


    def setup_source(self, source=""): 
        
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride if not isinstance(self.model, YOLO) else  self.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )

        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer
        )
        
        self.source_type = self.dataset.source_type

        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(YOLOStreamer.STREAM_WARNING)


    def setup_model(self, model, verbose=True, opt='autobackbone'):
        
        """Initialize YOLO model with given parameters and set it to evaluation mode if needed."""
        if self.model: 
            return self.model

        if opt == "autobackbone": 
            
            self.model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, verbose=verbose),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
                batch=self.args.batch,
                fuse=True,
                verbose=verbose,
            )

            self.device = self.model.device  # update device
            self.args.half = self.model.fp16  # update half
            self.model.eval()

        elif opt=="tracking": 
            self.track_history = defaultdict(list)
            cuda_use = torch.cuda.is_available()
            device = torch.device("cuda" if cuda_use else "cpu")
            self.model = YOLO(model)
            self.model = self.model.to(device)
            self.device = self.model.device
            self.stride = 32 


    def non_max_suppression(self, opt, detections,scores, conf, iou, thr):
        
        if len(detections)==0:
            warnings.warn("No Detections were applicable from the model...")
            return[]
        
        return operation.nms(detections, scores, iou_threshold=iou)            
        

    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue, *args, **kwargs):
        
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        with self._lock:  # for thread-safe inference
           
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            
            # Warmup model
            if not self.done_warmup and not isinstance(self.model, YOLO) :
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True
            else: 
                self.warmup(imgsz=(1, 3, *self.imgsz))
                self.done_warmup = True 

        
            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )

            self.run_callbacks("on_predict_start")
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                
                # Preprocess
                with profilers[0]:
                    images = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    if self.seen == 0:
                        with profile(activities=activities) as prof: 
                            preds = self.inference(images, *args, **kwargs)
                        prof.export_chrome_trace("trace_yolov8.json")
                    else: 
                        preds = self.inference(images, *args, **kwargs)

                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue
               
                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, images, im0s)

                if not isinstance(self.results[0], Results):
                    self.results = self.results[0]

                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)

                for i in range(n):
                    self.seen += 1
                    if isinstance(self.results[i], Results):
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                    else: 
                        self.speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                   
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), images, im0s, s)
                        if producer_flag is not None: 
                            producer_flag.value = True
                        if self.proc_image is not None and queue is not None:
                            queue.put(self.proc_image)
                        elif self.proc_image is None and queue is not None: 
                            queue.put(None)    
                        time.sleep(0.01)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *images.shape[2:])}" % t
            )
            
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")
    

    def translate_data(self, frame_index, video_path, images, results, orig_images)->None:
        
        """
        This function will only be needed in the case results are not of Type Results, 
        either use of AutoShape or AutoBackbone. 
        """
        
        names = self.model.names 
        orig_img = orig_images[frame_index]
        [height, width, _]= orig_img.shape
        shape = len(results.shape)
        try: 
            rectBoxes,class_ids = self.iteration_rows(results, frame_index, shape, height, width)
            if len(rectBoxes) == 0: 
                return []
            return Results(
                orig_img=orig_img,
                path=video_path, 
                names=names,
                boxes=rectBoxes,
                speed=self.speed,
                probs=class_ids
            )
        except KeyboardInterrupt as e: 
            exit(1)

      
    def write_results(self, i, p, im, original_images, s)->str:
        """Write inference results to a file or directory."""
        
        string = ""  # print string

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "%gx%g " % im.shape[2:]
        result = self.results[i] #Get the batch size pictures 

        if isinstance(result, torch.Tensor):
            result = self.translate_data(i, p, im, result, original_images)
            if not result: 
                return "(No Detection Found)"
        
        if self.mqtt_interface is not None:
            self.mqtt_interface.publish(self.mqtt_interface.topic, str(result.speed))
            time.sleep(0.01)

        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms" 
        self.points.clear()
        
        if result.boxes.id is not None and self.track_history is not None: 
            
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist() 
            clss = result.boxes.cls.cpu().tolist()

            current_track_ids = set()
            
            for box, track_id, cls in zip(boxes, track_ids,  clss):
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 
                current_track_ids.add(track_id)

                if track_id not in self.track_history or len(self.track_history[track_id])==0:
                    self.track_history[track_id] = [bbox_center]
                else:
                    self.track_history[track_id].append(bbox_center)    
                
                if len(self.track_history) > 30:
                    self.track_history.pop(list(self.track_history.keys())[0])
                
                track = self.track_history[track_id]
                self.points[cls] = np.hstack(track).astype(np.int32).reshape((-1,1,2))
                
                # track.append((float(bbox_center[0]), float(bbox_center[1])))
                
            #Remove track IDs from track history that were not detected in the current frame 
            lost_track_ids = set(self.track_history.keys()) - current_track_ids
            for lost_track_id in lost_track_ids:
                self.track_history.pop(lost_track_id, None)
                self.points = {cls:pts for cls, pts in self.points.items() if cls not in lost_track_ids}
                
            for region in self.regions:
                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                    region["counts"] += 1
        
        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        
        if self.args.show:
            self.show(str(p))
        
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string
    

    def save_predicted_images(self, save_path="", frame=0):
        return super().save_predicted_images(save_path, frame)
    

    def show(self, p=""): 
        return super().show(p)
    

    def run_callbacks(self, event): 
        return super().run_callbacks(event)
    

    def add_callback(self, event, func):
        return super().add_callback(event, func)


    def calculate_padding(self, original_height, original_width, target_dimension):
        
        # Calculate the scale needed to fit the width and height into the target dimension
        # Determine which scale to use (the one that fits the image entirely within the target box)
        scale_used = min(target_dimension / original_width, target_dimension / original_height)

        # Calculate the effective width and height after scaling
        # Calculate padding by subtracting the effective dimensions from the target dimension
        effective_width = original_width * scale_used
        effective_height = original_height * scale_used
        
        return target_dimension - effective_width, target_dimension - effective_height, scale_used
    

    def box_creation(self, results, i, r, shape): 
        
        if shape == 3: 
            return [results[i, 0, r] - results[i, 2, r]/2,
                    results[i, 1, r] - results[i, 3, r]/2,
                    results[i, 0, r] + results[i, 2, r],
                    results[i, 1, r] + results[i, 3, r]]
        
        return [results[0, r] - results[2, r]/2,
                results[1, r] - results[3, r]/2,
                results[0, r] + results[2, r],
                results[1, r] + results[3, r]]


    def scale_boxes(self, boxes, pad_x, pad_y, scale): 
        boxes[:,[0,2]] -= pad_x // 2
        boxes[:,[1,3]] -= pad_y // 2 
        boxes[:, :4] /= scale
        return boxes 
    

    def data_to_tensor_filter(self, boxes, scores, class_ids): 
        boxes = torch.FloatTensor(boxes)
        scores = torch.FloatTensor(scores)
        class_ids = torch.LongTensor(class_ids)
        result_boxes = self.non_max_suppression(opt=None, detections=boxes,scores=scores, conf=0.25, iou=0.45, thr=0.5)
        if len(result_boxes)==0 or len(boxes)==0: 
                return [],[],[]
        return boxes[result_boxes], scores[result_boxes], class_ids[result_boxes]
        

    def class_scores_creation(self, results, i, r, shape): 
        if shape == 3:
            return results[i, 4:, r]
        return results[4:, r]


    def iteration_rows(self, results, frame_index, shape, height, width, conf_thr=0.4): 
        boxes, class_ids, scores = [],[],[]

        pad_x,pad_y,scale = self.calculate_padding(height,width,640)

        for r in range(results.shape[-1]):
            classes_scores = self.class_scores_creation(results, frame_index, r, shape)
            
            (_, maxScore, _, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores.cpu().numpy()) 
            if maxScore >= conf_thr:
               boxes.append(self.box_creation(results, frame_index, r, shape))
               class_ids.append(maxClassIndex)
               scores.append(maxScore)
        try: 
            boxes, scores, class_ids = self.data_to_tensor_filter(boxes=boxes, scores=scores, class_ids=class_ids)
            boxes = self.scale_boxes(boxes, pad_x=pad_x, pad_y=pad_y, scale=scale)
            return torch.stack((boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],scores, class_ids), axis=-1), class_ids

        except:
            boxes, scores,class_ids  = [],[],[]
            return torch.empty((0,6)),class_ids
        

    def count_regions(self) -> list:
        return super().count_regions()
    
    
    def mouse_callback(self, event, x, y, flags, param) -> None:
        return super().mouse_callback(event, x, y, flags, param)
    

        

   
