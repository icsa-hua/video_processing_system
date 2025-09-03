from obs_system.compressed.interface.compressed_yolo import CompressedYOLO 
from obs_system.detection_module.interface.streaming import YOLOStreamer
from obs_system.utils.tiles import *
from obs_system.utils.appraisal import StepContext
from obs_system.utils.logger import logger 

import os 
import re
import pdb
import torch
import cv2
import time
import numpy as np 
import supervision as sv
import torchvision.ops as operation 
 
from typing import Any, List
from queue import Queue 
from abc import ABC, abstractmethod 
from pathlib import Path 
from ultralytics import YOLO
from torch.profiler import profile, ProfilerActivity 
#from trackers import SORTTracker 
from torch.nn.utils.rnn import pad_sequence 
#from trackers.core.deepsort.tracker import DeepSORTTracker 
from collections import defaultdict, Counter
from ultralytics.utils import DEFAULT_CFG, ops, callbacks
from ultralytics.engine.results import Results
from ultralytics.utils.files import increment_path 
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import colorstr 


def get_frame_ids(labels:List[str])->List[str]: 

    frame_ids = [value.split(' ') for value in labels]
    frame_ids = [id[3] for id in frame_ids]  
    frame_ids = [re.sub(r'[^\w]','|', id) for id in frame_ids]
    frame_ids = [id.split('|') for id in frame_ids]
    frame_ids = [id[0] for id in frame_ids] 
    return frame_ids


class OnnxY8Streamer(YOLOStreamer): 

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg,overrides, _callbacks)

        self.color = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", 
            "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66"
            ])

        self.tracker = None 
        self.tracker_choice = 'byte_tracker' 
        self.box_annotator = sv.BoxAnnotator(color=self.color, color_lookup=sv.ColorLookup.TRACK) 
        self.source = ""
        self.CONFIDENCE_THRESHOLD = 0.5 
        self.NMS_THRESHOLD = 0.4 

    
    def warmup(self, imgsz=(1,3,640,640)): 
        return super().warmup(imgsz)


    def from_numpy(self, x: np.ndarray):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    def __call__(self, source=None, model=None, logic_module=None, mqtt_broker=None,producer_flag=None, queue=None, *args, **kwargs): 

        self.mqtt_interface = mqtt_broker 
        self.args.stream_buffer = True 
        self.logic_module = logic_module 

        try: 
            self.predict_cli(source=os.path.normpath(os.path.abspath(source)) if os.path.isfile(source) else source, 
                model=model, 
                producer_flag=producer_flag, 
                queue=queue
            )

        except KeyboardInterrupt as ke: 
            if producer_flag is not None: 
                producer_flag.value=False 
            if self.logic_module is not None and self.logic_module["DAV2"] is not None: 
                self.logic_module["DAV2"].deallocate_resources() 
            cv2.destroyAllWindows() 
            logger.exception(f"KeyboardInterrupt: {ke}")
        
        return 


    def pre_transform(self, im): 
        return super().pre_transform(im) 


    def inference(self, im, orig_images, *args, **kwargs): 
        image_boxes, image_scores, image_class_ids = self.model(im) 
        if len(image_boxes) == 0: return [] 
        #Boxes and the others here are lists with 16 length(for each image) 

        preds = [] 
        pad_x, pad_y, scale = self.converter.calculate_padding(self.orig_height, self.orig_width, 640)
        for image in range(len(image_boxes)): 
        
            boxes, scores, class_ids = self.converter.data_to_tensor_filter(boxes=image_boxes[image], scores=image_scores[image], class_ids=image_class_ids[image]) 
                
            if len(boxes) != 0: 
                boxes = self.converter.scale_boxes(
                    boxes=boxes, 
                    pad_x=pad_x,
                    pad_y=pad_y, 
                    scale=scale
                )

                #padded_boxes = pad_sequence(sequences=boxes, batch_first=true, padding_value=float(0))  
                inf_results = torch.stack(
                        (boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], scores, class_ids),
                        axis=-1
                )

                results = results(
                    orig_img = orig_images[image], 
                    path = f"image{image}.jpg", 
                    names = self.converter.class_names, 
                    boxes = inf_results, 
                    speed = {}, 
                    probs = class_ids
                ) 

                detections = sv.detections.from_ultralytics(results) 
                if self.tracker_choice == 'sort': 
                    detections = self.tracker.update(detections) 
                elif self.tracker_choice == 'deepsort': 
                    detections = self.tracker.update(detections, orig_images[image]) 
                elif self.tracker_choice == 'byte_tracker': 
                    detections = self.tracker.update_with_detections(detections) 
                

                detections = detections[detections.tracker_id != -1] 
                del results
                del inf_results 
                results = self.build_results_from_detections(detections, orig_images[image], image)
                preds.append(results)
            else: 

                
                results = results(
                    orig_img = orig_image, 
                    path = self.source, 
                    names = self.converter.class_names, 
                    boxes = [], 
                    speed = {}, 
                    probs = []
                )
                preds.append(results)

        return preds


    def build_results_from_detections(self, detections, orig_image, image): 

        if len(detections) == 0: 

            results = Results(
               orig_img=orig_image, 
               path=self.source, 
               names=self.converter.class_names, 
               boxes=torch.zeros((0,6),dtype=torch.float32), 
               speed={}, 
            )
            return results 


        xyxy = torch.from_numpy(detections.xyxy).to(torch.float32)
        scores_t = torch.from_numpy(detections.confidence).to(torch.float32) 
        class_t = torch.from_numpy(detections.class_id).to(torch.float32) 
        ids_t = torch.from_numpy(detections.tracker_id).to(torch.float32) 
        inf_results = torch.stack((xyxy[:,0],xyxy[:,1],xyxy[:,2],xyxy[:,3],
            ids_t.view(-1),
            scores_t.view(-1), 
            class_t.view(-1), 
        ),  axis=-1)

        results = Results(
            orig_img=orig_image, 
            path=f"image{image}.jpg", 
            names=self.converter.class_names, 
            boxes=inf_results, 
            speed={}, 
            probs=class_t
        )

        return results


    def preprocess(self,im): 
        """
        Prepare input image before inference. 

        Args: 
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list. 
        """

        not_tensor = not isinstance(im, torch.Tensor) 

        if not_tensor: 
            if isinstance(im, np.ndarray): im = list(im)
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im) 
            im = torch.from_numpy(im) 

        im = im.to(self.device) 

        if (not isinstance(self.model, YOLO) and not isinstance(self.model, CompressedYOLO)): 
            im = im.half() if self.model.fp16 else im.float() 

        else: 
            im = im.float() 

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0

        #from torchvision.utils import save_image
        #save_image(im,'debug.png')

        return im


    def non_max_suppression(self,detections,score,iou): 
        return super().non_max_suppression(detections, score, iou)


    def postprocess(self, preds, img, orig_img): 
        return super().postprocess(preds, img, orig_img) 


    def predict_cli(self, source, model, producer_flag=None, queue=None): 
        return super().predict_cli(source, model, producer_flag, queue) 


    def setup_source(self, source=""):
        super().setup_source(source) 


    def check_onnx_model(self, model_path, device):
        if not os.path.exists(model_path):
            model = YOLO('yolov8s.pt') 
            #model.export(
            #    format="onnx",
            #    imgsz=(640, 640),
            #    opset=12,
            #    simplify=True,
            #    dynamic=True,
            #    half=True,
            #    device=device,
            #    #batch=16, 
            #    #name="yolov8n_640_dynamic_cpu_fp16_simplified_op12",
            #     )
            model.export(
                format="onnx",
                imgsz=(640), 
                dynamic=True,
                simplify=True
            )
    

    def setup_model(self, model, verbose=True, opt=''): 

        device = select_device(self.args.device, verbose=verbose) 
        model_path = 'obs_system/compressed/yolov8s.onnx'
        self.check_onnx_model(model_path, device)  
        self.model = CompressedYOLO(model_path) 
        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        if self.tracker_choice == 'sort': 
            self.tracker = SORTTracker() 
        elif self.tracker_choice == 'deepsort': 
            # NOTE: This needs some adjustments after trackers latest update. 
            self.tracker = DeepSORTTracker()
        elif self.tracker_choice == 'byte_tracker': 
            self.tracker = sv.ByteTrack() 
        else: 
            self.tracker_choice = 'sort' 
            self.tracker = SORTTracker() 

        self.tracker.reset()  
        self.track_history = defaultdict(list)
        self.stride = 32 
        self.args.half = 16        


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue, *args, **kwargs): 

        if self.args.verbose: logger.info("") 
        self.source = source 
        
        with self._lock: 

            self.setup_source(source if source is not None else self.args.source)
            


            for batch in self.dataset: 
                paths, im0s, s = batch 
                if self.logic_module is not None and self.logic_module["ROI"] is not None: 
                    self.logic_module["ROI"].set_regions(im0s[0]) 
                break 

            self.orig_height,self.orig_width = im0s[0].shape[:2]
            if self.args.save or self.args.save_txt: 
                (self.save_dir / "labels" if self.args.save_txt else  self.save_dir).mkdir(parents=True) 

            self.seen, self.windows, self.batch = 0, [], None 
            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device)
            )

            self.run_callbacks("on_predict_start") 
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

            for self.batch in self.dataset: 
                self.run_callbacks("on_predict_batch_start") 
                paths, im0s, s = self.batch 
                tiles_batch = []
                with StepContext(name="Batch Tiles", catch=(RuntimeError,)):
                    #self.process_tiles(im0s, s)     
                    
                    frame_ids = get_frame_ids(labels=s) 
                    for i,image in enumerate(im0s): 
                        tiles = split_image(image,frame_id=frame_ids[i], tile_size=640, show_tiles=False, overlap=0.15) 
                        tiles_batch.append(tiles)
                    self.microbatched(tiles_batch, im0s, self.device, micro=32) 
                import pdb;pdb.set_trace()

                if self.logic_module is not None and self.logic_module["ROI"] is not None: 
                    im0s = self.logic_module["ROI"].crop_image(im0s) 

                motion_flags = self.logic_module["SUBTRACTOR"].detect(im0s, threshold=500) 

                filtered_indices = [i for i, m in enumerate(motion_flags) if m] 

                if not filtered_indices: continue 

                paths = [paths[i] for i in filtered_indices] 
                s = [s[i] for i in filtered_indices] 
                tmp_im0s = [im0s[i] for i in filtered_indices] 

                import pdb; pdb.set_trace()
                # Create the frame queue for tile maker. 
                

                with profilers[0]: 
                    images = self.preprocess(tmp_im0s) 

                with profilers[1]: 
                    if self.seen == 0: 
                        with profile(activities=activities) as prof: 
                            preds = self.inference(images, im0s, *args, **kwargs)
                        prof.export_chrome_trace(f"trace_{model}.json") 

                    else: 
                        preds = self.inference(images, im0s, *args, **kwargs) 

                    if self.args.embed: 
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds 
                        continue 

                with profilers[2]: 
                    self.results = self.postprocess(preds, images, im0s) 
                

                if not isinstance(self.results[0], Results): 
                    self.results = self.results[0]
                    self.results = torch.reshape(self.results, self.results.shape[0], self.results.shape[2], self.results.shape[1]) 
                
                self.run_callbacks("on_predict_postprocess_end") 

                n = len(images) 

                if self.logic_module is not None and self.logic_module["DAV2"] is not None: 
                    fps = self.dataset.fps if self.dataset.mode == "video" else 30 
                    self.logic_module["DAV2"].dataset(images, fps, "runs/detect/DI_results/dav2_detections") 

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

                        time.sleep(0.08) 

                    self.capture_object_boxes(i,im0s[i], self.results[i], cropped_dirname=self.cropped_image_dirname)

                if self.args.verbose: 
                    logger.info("".join(s)) 

                self.run_callbacks("on_predict_batch_end") 
                yield from self.results 
        
        for v in self.vid_writer.values(): 
            if isinstance(v, cv2.VideoWriter): 
                v.release() 

        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            logger.info(
                    f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                    f"{(min(self.args.batch, self.seen), 3, *images.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop: 
            nl = len(list(self.save_dir.glob("labels/*.txt"))) 
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}"
            logger.info(f"Results saves to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")  
        yield from self.results 


    def write_results(self, i, p, im, original_images, s)->str: 
       return super().write_results(i, p, im, original_images, s) 


    def empty_Results_instance(self,orig_image): 
        return Results(
            orig_img=orig_image,
            path=self.source, 
            names=self.converter.class_names,
            boxes=torch.zeros((0,6),dtype=torch.float32), 
            speed={}
        )



    def process_tiles(self, im0s, s): 
        
        frame_ids = get_frame_ids(labels=s) 

        tile = self.height if self.height == self.width else 640 
        overlap = 0.15 

        in_queue = Queue(maxsize=100) 

        for frame, f_id in zip(im0s, frame_ids): 
            H, W = frame.shape[:2]

            for (x0, y0, tw, th) in make_tiles(W, H, tile, overlap): 
                slice = frame[y0:y0+th, x0:x0+tw] 

                if slice.shape[0] != th or slice.shape[1] != tw: 
                    slice = cv2.copyMakeBorder(
                        slice, 
                        0, th - slice.shape[0], 0, tw-slice.shape[1],
                        cv2.BORDER_REPLICATE
                    )

                crop, scale, pad = letterbox(
                    img=slice, 
                    new_shape=(tile, tile)
                )

                meta_data = {
                    "frame_id": f_id, 
                    "offset":(x0,y0), 
                    "scale":scale, 
                    "pad": pad, 
                    "orig_shape":(H, W, 3)
                }

                in_queue.put((crop, meta_data))


    def microbatched(self, tiles_batch, orig_images, device, micro=32): 

        acc = defaultdict(list) 
        seen = Counter() 
        expected = {} 

        stream = flatten_tiles(tiles_batch) 

        # Every tile inside the stream looks correct / RGB image 460x640 in resolution 
#        for tile in stream: 
#            cv2.imwrite(filename=f"tile_{tile[1]['t_idx']}.jpg", img=tile[0])
#            break

        # Double buffers in pinned host memory. 
        host0 = np.empty((micro,640, 640, 3), dtype=np.uint8) 
        host1 = np.empty_like(host0) 

        metas0, metas1 = [None]*micro, [None]*micro 

        def fill(host, metas): 
            n = 0 
            try: 
                for n in range(micro): 
                    tile, meta = next(stream)
                    host[n][:] = tile 
                    metas[n] = meta 
            except StopIteration: 
                pass 

            return n 

        # Pre-fill first 
        n0 = fill(host0, metas0) 
        if n0 == 0: return {} 

        images = self.preprocess(host0) 
        image_boxes, image_scores, image_class_ids = self.model(images)
        pdb.set_trace() 
        host0 = np.transpose(host0, (0,3,1,2))

        # Host host0 has all the tiles from the stream. 
        #images = self.preprocess(host0)
        #image_boxes, image_scores, image_class_ids = self.model(host0)
        #import pdb;pdb.set_trace()

        tbuf = torch.empty((micro, 3, 640, 640), device=device, dtype=torch.float32) 

        cur_host, cur_metas, cur_n = host0, metas0, n0 
        next_host, next_metas = host1, metas1 

        while cur_n > 0: 
            #Prefetch next on CPU while GPU runs (simple overlap)
            n1 = fill(next_host, next_metas) 
            tb = torch.from_numpy(cur_host[:cur_n]).to(device, non_blocking=True) 
            tb = tb.permute(0,3,1,2 ).to(dtype=torch.float32) 
            tb.mul_(1.0/255.0) 
            tbuf[:cur_n].copy_(tb, non_blocking=True) 
    
            image_boxes, image_scores, image_class_ids = self.model(tbuf[:cur_n])

            


def flatten_tiles(tiles_batch): 
    for fb in tiles_batch: 
        fid = fb[0][1]['frame_id']
        for t_idx, (tile, meta) in enumerate(fb): 
            yield tile, {"frame_id":fid, "t_idx":t_idx, **meta}





























        
