from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.streaming import YOLOStreamer
from obs_system.logic_module.dummy_logic.tracker_sv import TrackerHandler
from obs_system.utils.tiles import *
from obs_system.utils.appraisal import StepContext
from obs_system.utils.logger import get_logger 

import os 
import gc
import pdb
import time 
import torch
import cv2
import numpy as np 
import supervision as sv

from typing import Any
from memory_profiler import profile as mem_profile
from pathlib import Path 
from ultralytics import YOLO
from torch.profiler import profile, ProfilerActivity 
from torchvision.ops import batched_nms, nms
from collections import defaultdict, deque
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import colorstr 


logger = get_logger("obs_system."+__name__)


class TensorRTRTXStreamer(YOLOStreamer): 

    def __init__(self, cfg:Any=DEFAULT_CFG, overrides=None, _callbacks=None)->None: 
        super().__init__(cfg,overrides, _callbacks)
        self.source = ""


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
        pass 


    def preprocess(self,im:Any): 
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

        if (not isinstance(self.model, YOLO) and not isinstance(self.model, TensorRTYOLO )): 
            im = im.half() if self.model.fp16 else im.float() 

        else: 
            im = im.float() 

        if not_tensor:
            im = im.div(255.0)  # 0 - 255 to 0.0 - 1.0

        #from torchvision.utils import save_image
        #save_image(im,'debug.png')

        return im



    def non_max_suppression(self,detections,scores,iou): 
        return super().non_max_suppression(detections, scores, iou)


    def postprocess(self, preds, img, orig_imgs): 
        return super().postprocess(preds, img, orig_imgs) 


    def predict_cli(self, source, model, producer_flag=None, queue=None): 
        return super().predict_cli(source, model, producer_flag, queue) 


    def setup_source(self, source=""):
        super().setup_source(source) 


    def setup_model(self, model, opt='tracking'): 

        device = select_device(self.args.device, verbose=self.args.verbose) 

        model_path = 'obs_system/compressed/yolov8s.onnx'

        self.model = TensorRTYOLO(engine_path=model_path)

        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        self.tracker_model = TrackerHandler(tracker_choice="byte_tracker") if opt == "tracking" else None 

        self.stride = 32 if not self.args.half else 16

        if self.args.verbose: 
            logger.info(f"[checked] Model {model} successfully set up")


    def write_results(self, i, p, im, original_images, s)->str: 
       return super().write_results(i, p, im, original_images, s) 


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue, *args, **kwargs): 
        return self._stream_inference_impl(source, model, producer_flag, queue, *args, **kwargs)


    @mem_profile
    def _stream_inference_impl(self, source, model, producer_flag, queue, *args, **kwargs): 
        if self.args.verbose: logger.info(" ") 
        
        self.source = source 
        with self._lock: 
            self.setup_source(source if source is not None else self.args.source)
    
            self.frame_images = {} 
            micro = 32
            overlap_ratio = 0.15
            self.seen = 0 
            self.windows = [] 
            self.batch = None 
            self.results = []
            start_time = time.perf_counter() 

            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
            )
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] 
            
            self.run_callbacks("on_predict_start") 

            # Warmup for Better Inference. Reduces Initial frames high inference time and is more stable. 
            with StepContext(name="Warmup Session", catch=(Exception, RuntimeError), verbose=self.args.verbose):
                if not self.done_warmup: 
                    self.model.warmup(micro=micro, warmup_sessions=4)
                    self.done_warmup = True

            use_roi = True if self.logic_module is not None and self.logic_module["ROI"] is not None else False 

            host0 = np.empty((micro, self.imgsz[0], self.imgsz[1], 3), np.uint8) 
            metas0 = [None]*micro 
            tbuf = torch.empty((micro, 3, self.imgsz[0], self.imgsz[1]), device=self.device, dtype=torch.float32)
            
            # Generators 
            frame_iter = self.iter_data(use_roi=use_roi)
            tile_stream = self._frames_to_tiles(frame_iter, tile_size=self.imgsz[0], overlap_ratio=overlap_ratio)

            pending = {} 
            while True: 
                n0 = next_microbatch(tile_stream, micro, host0, metas0)
                if n0 == 0: 
                    break 
                
                if not self.batch or not self.batch[1]: 
                    break 

                if self.seen >= len(self.batch[1]): 
                    self.seen = 0 
                    self.results.clear()

                with profilers[0]:
                    tb = (self.preprocess(host0[:n0])).to(self.device, non_blocking=True) 
                    tbuf[:n0].copy_(tb, non_blocking=True) 

                with profilers[1]:
                    if self.seen == 0 and self.args.verbose: 
                        with profile(activities=activities) as prof:
                            i_boxes, i_scores, i_classes = self.model(tbuf[:n0], debug=self.args.verbose) 
                        prof.export_chrome_trace(f"trace_{model}.json")
                    else: 
                        i_boxes, i_scores, i_classes = self.model(tbuf[:n0], debug=self.args.verbose)

                with profilers[2]: 
                    pass 

                with StepContext(name="Post Process", catch=(Exception, RuntimeError), verbose=self.args.verbose): 
                    frames_out = {} 
                    for det, score, cls_, meta in zip(i_boxes, i_scores, i_classes, metas0[:n0]): 
                        if meta is None:
                            continue 

                        f_id = int(meta["frame_id"])
                        s = pending.setdefault(
                            f_id, 
                            {"need":(meta['grid'][0]*meta['grid'][1]), 
                             "seen":set(), 
                             "parts":[]
                            }
                        )
                        s['seen'].add(meta['t_idx'])

                        if det is None or len(det) == 0:
                            mapped_boxes = np.empty((0,4), np.float32) 
                            score = np.empty((0,), np.float32) 
                            cls_ = np.empty((0,), np.int64) 

                        else: 
                            mapped_boxes = reconstruct_tiles(
                                boxes_xyxy=det, 
                                tx=meta['left_x'], 
                                ty=meta['top_y'], 
                                orig_H=meta['f_wh'][0], 
                                orig_W=meta['f_wh'][1], 
                                gain=meta['gain'], 
                                pad=(meta['pad_x'],meta['pad_y'])
                            )
                            mapped_boxes = mapped_boxes.cpu().numpy() if isinstance(mapped_boxes,torch.Tensor) else mapped_boxes
                            score = score.cpu().numpy() if isinstance(score, torch.Tensor) else score
                            cls_ = cls_.cpu().numpy() if isinstance(cls_, torch.Tensor) else cls_
                        s["parts"].append((mapped_boxes, score, cls_)) 

                        if len(s["seen"]) == s["need"]: 
                            if not s["parts"] or all(p[0].shape[0] == 0 for p in s["parts"]): 
                                B = np.empty((0,4), np.float32) 
                                S = np.empty((0,), np.float32) 
                                C = np.empty((0,), np.float32) 

                            else: 
                                B = np.concatenate([p[0] for p in s["parts"]], 0)
                                S = np.concatenate([p[1] for p in s["parts"]], 0) 
                                C = np.concatenate([p[2] for p in s["parts"]], 0)               

                            frames_out[f_id] = (B,S,C) 
                            del pending[f_id]

                    del i_boxes
                    del i_scores 
                    del i_classes 
                    
                    for f_id,(boxes, scores, classes) in frames_out.items(): 
                        
                        if self.seen >= len(self.batch[1]): 
                            self.seen = 0
                        if isinstance(boxes, np.ndarray): 
                            boxes_t = torch.from_numpy(boxes)
                            scores_t = torch.from_numpy(scores) 
                            classes_t = torch.from_numpy(classes) 
                        else: 
                            boxes_t = boxes 
                            scores_t = scores 
                            classes_t = classes 

                        keep_pc = batched_nms(
                            boxes_t, 
                            scores_t, 
                            classes_t.long(), 
                            iou_threshold=(1.0-self.args.iou)
                        )

                        keep = keep_pc if keep_pc.numel()==0 else keep_pc[nms(boxes_t[keep_pc],scores_t[keep_pc], iou_threshold=(1-self.args.iou))]
                        boxes_t, scores_t, classes_t = boxes_t[keep], scores_t[keep], classes_t[keep] 

                        inf_results = torch.stack(
                            (
                                boxes_t[:,0], 
                                boxes_t[:,1],
                                boxes_t[:,2],
                                boxes_t[:,3],
                                scores_t, 
                                classes_t
                            )
                        )

                        results = Results(
                            orig_img=self.frame_images[f_id], 
                            path=f"image_{f_id}.jpg", 
                            names=self.converter.class_names, 
                            boxes=inf_results.T, 
                            speed={}, 
                        )

                        if self.tracker_model is not None:
                            results = self.tracker_model.detect(
                                predictions=results, 
                                save=False, 
                                orig_frame=self.frame_images[f_id], 
                                f_id=f_id, 
                                class_names=self.converter.class_names
                            )
                        
                        self.results.append(results)                                             
                        self.frame_images.pop(f_id,None)
                        del inf_results 
                        yield results

                        bn = len(frames_out.keys())
                        self.results[self.seen].speed = {
                                "preprocess": profilers[0].dt * 1e3 / bn,
                                "inference": profilers[1].dt * 1e3 / bn, 
                                "postprocess": profilers[2].dt * 1e3 / bn
                        }
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show: 
                            filename=Path(self.batch[0][self.seen])
                            if not filename: 
                                logger.warning("[WARNING]: filename to save image is invalid")
                            
                            self.batch[2][self.seen] += self.write_results(
                                i = self.seen, 
                                p = filename,  
                                im= tbuf[:n0],
                                original_images=self.batch[1], 
                                s = self.batch[2]
                            )
                        if producer_flag is not None : 
                           producer_flag.value = True 

                        if self.proc_image is not None and queue is not None: 
                            queue.put(self.proc_image) 
                        elif self.proc_image is None and queue is not None: 
                            queue.put(None) 

                        with StepContext(name="Crop Objects to Image", catch=(RuntimeError,), verbose=self.args.verbose):
                            try: 
                                self.capture_object_boxes(
                                        image=self.batch[1][self.seen],
                                        results=self.results[self.seen],
                                        cropped_dirname=self.cropped_image_dirname,
                                        save=self.args.save
                                )
                            except IndexError as ie: 
                                logger.exception(ie)

                        self.seen += 1 
                        if self.seen >= len(self.batch[1]): 
                            self.seen = 0
                            self.results.clear()

                        if self.seen == len(self.batch)-1 and self.args.verbose: 
                            elapsed_time = time.perf_counter() - start_time
                            logger.info(f"Time From Capturing batch to meaningfull inference is {elapsed_time:.2f}")

                self.run_callbacks("on_predict_postprocess_end")
                self.run_callbacks("on_predict_batch_end")

        for v in self.vid_writer.values(): 
            if isinstance(v, cv2.VideoWriter): 
                v.release() 

        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *tbuf.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")        



    @mem_profile
    def _stream_inference_impl_2(self, source, model, producer_flag, queue, *args, **kwargs): 

        if self.args.verbose: logger.debug("") 

        self.source = source
        with self._lock: 
            self.setup_source(source if source is not None else self.args.source) # Create Dataset obj. 
            
            dtype_f = torch.float16 if self.args.half else torch.float32
            dtype_i = torch.int32 if self.args.half else torch.int64
            pending = {} 
            frames_images = {}
            max_f_inflight = 16 # Frames batch size 
            micro = 32 # Maximum number for tiles to exist. 
            f_inflight = set() 
            frames_queue = deque()
            tile_queue = deque()

            def enqueue_frame(frame_id, img): 
                tiles = split_image(is_tensor=False, image=img, frame_id=frame_id, tile_size=self.imgsz[0], overlap=0.15)
            
                for i, (t, m) in enumerate(tiles): 
                    m['t_idx'] = i
                    s = pending.setdefault(
                    frame_id, 
                        {"need":(m['grid'][0]*m['grid'][1]), 
                         "seen":set(), 
                         "parts":[] 
                        }
                    )
                    
                    tile_queue.append((t, m)) # Fill the tile_queue
                f_inflight.add(frame_id) # Fill the incoming frames 
                frames_images[frame_id] = img # Map frame id with frame. 

            out_dir="assets/save_inferences/"
            self.seen = 0 
            self.windows = [] 
            self.batch = None
            use_roi = False

            profilers = (
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
                ops.Profile(device=self.device), 
            )
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] 

            self.run_callbacks("on_predict_start") 

            # Warmup for Better Inference. Reduces Initial frames high inference time and is more stable. 
            with StepContext(name="Warmup Session", catch=(Exception, RuntimeError)):
                if not self.done_warmup: 
                    self.model.warmup(micro=micro, dtype=dtype_f, warmup_sessions=4)
                    self.done_warmup = True

            if self.args.save or self.args.save_txt: 
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True)

            # Do we use regions of interest to crop the frames? 
            if self.logic_module is not None and self.logic_module["ROI"] is not None: 
                use_roi = True 

            self.dataset = iter(self.dataset)
            self.results = []

            # Fill the frames Queue after checking motion on frames. 
            frames_queue = self.admit_frames(
                fr_queue = frames_queue, 
                max_f_inf = max_f_inflight, 
                use_roi = use_roi
            )

            # Initialize the buffers to host the tiles and metas. 
            host0 = np.empty((micro, self.imgsz[0], self.imgsz[1], 3), np.uint8) 
            host1 = np.empty_like(host0) 

            metas0, metas1 = [None]*micro, [None]*micro 
            tbuf = torch.empty((micro, 3, self.imgsz[0], self.imgsz[1]), device = self.device, dtype=dtype_f)

            # Helper function to refill the tile_queue, frames inflight and remove the frames from the frame queue. 
            def refill(): 
                while len(tile_queue)< micro and len(f_inflight) < max_f_inflight and frames_queue: 
                    #Frame queue starts at max size and steadily decreases. 
                    f_id, im = frames_queue.popleft()
                    enqueue_frame(f_id, im) 

            refill()

            # Main loop that executes until no tiles or frames are left pending 
            while tile_queue or pending or frames_queue: 

                if len(tile_queue) < (micro //2) and len(f_inflight) < max_f_inflight: 

                    # When tiles are low bring the next dataset if frames_queue is low though. 
                    frames_queue = self.admit_frames(
                        fr_queue=frames_queue, 
                        max_f_inf=max_f_inflight, 
                        use_roi=use_roi
                    )

                    refill() 

                n0, host0, metas0, tile_queue = fill(
                    host=host0, 
                    metas=metas0, 
                    tile_queue=tile_queue,
                    micro=micro
                )

                if n0 == 0: continue

                cur_host, cur_metas = host0, metas0 
                nxt_host, nxt_metas = host1, metas1 

                n1, nxt_host, nxt_metas, tile_queue = fill(
                    host=nxt_host, 
                    metas=nxt_metas, 
                    tile_queue=tile_queue, 
                    micro=micro
                )

                with profilers[0]:
                    tb = (self.preprocess(cur_host[:n0])).to(self.device, non_blocking=True) 
                    # tb = torch.permute(tb,[0,3,1,2]).to(dtype=dtype_f)  
                    tb.mul(1.0/255.0)
                    tbuf[:n0].copy_(tb, non_blocking=True) 

                with profilers[1]:
                    if self.seen == 0: 
                        with profile(activities=activities) as prof:
                            i_boxes, i_scores, i_classes = self.model(tbuf[:n0], dtype_f, dtype_i) 
                        prof.export_chrome_trace(f"trace_{model}.json")
                    else: 
                        i_boxes, i_scores, i_classes = self.model(tbuf[:n0])

                with profilers[2]:
                    pass 

                with StepContext(name="Post Process", catch=(Exception, RuntimeError), verbose=True): 

                    frames_out = {} 
                    for det, score, cls_, meta in zip(i_boxes, i_scores, i_classes, cur_metas[:n0]): 
                        
                        if meta is None: continue 
                    
                        f_id = int(meta['frame_id'])
                        s = pending.setdefault(
                            f_id, 
                            {"need":(meta['grid'][0]*meta['grid'][1]), 
                             "seen":set(), 
                             "parts":[] 
                            }
                        )

                        s["seen"].add(meta['t_idx'])

                        if det is None or len(det) == 0: 
                            mapped_boxes = torch.empty((0,4), dtype=dtype_f) 
                            score = torch.empty((0,), dtype=dtype_f) 
                            cls_ = torch.empty((0,), dtype=dtype_i) 

                        else: 
                            mapped_boxes = reconstruct_tiles(
                                boxes_xyxy=det, 
                                tx=meta['left_x'], 
                                ty=meta['top_y'], 
                                orig_H=meta['f_wh'][0], 
                                orig_W=meta['f_wh'][1], 
                                gain=meta['gain'], 
                                pad=(meta['pad_x'],meta['pad_y'])
                            )

                        s["parts"].append( (mapped_boxes, score, cls_) ) 

                        if len(s["seen"]) == s["need"]: 

                            if len(s["parts"]) == 0 or all(p[0].shape[0] == 0 for p in s["parts"]): 
                                frames_out[f_id] = (
                                    torch.empty((0,4), dtype=dtype_f), 
                                    torch.empty((0,), dtype=dtype_f), 
                                    torch.empty((0,), dtype=dtype_i)
                                ) 

                            else: 
                                frames_out[f_id] = (
                                    torch.concatenate([p[0] for p in s["parts"]], 0),
                                    torch.concatenate([p[1] for p in s["parts"]], 0), 
                                    torch.concatenate([p[2] for p in s["parts"]], 0)
                                )

                            del pending[f_id] 
                        
                        del mapped_boxes 
                        del score 
                        del cls_


                    del i_boxes
                    del i_classes
                    del i_scores
                    gc.collect()

                    for f_id in (frames_out.keys()):
                        out_image = frames_images.get(f_id)
                        if out_image is None: continue
                        boxes_t, scores_t, classes_t = frames_out[f_id]
                        # boxes_t = torch.from_numpy(boxes)
                        # scores_t = torch.from_numpy(scores) 
                        # classes_t = torch.from_numpy(classes) 

                        if boxes_t is None or len(boxes_t) == 0: 
                            cv2.imwrite(os.path.join(out_dir, f"{f_id}.jpg"), out_image)
                            continue

                        keep_pc = batched_nms(boxes_t, scores_t, classes_t.long(), iou_threshold=(1-self.args.iou))

                        if keep_pc.numel() == 0:
                            keep_final = keep_pc  # empty
                        else:
                            boxes_pc  = boxes_t[keep_pc]
                            scores_pc = scores_t[keep_pc]

                            keep_ca = nms(boxes_pc, scores_pc, iou_threshold=self.args.iou)

                            # 3) map back to original indices
                            keep_final = keep_pc[keep_ca]

                        boxes, scores, classes = boxes_t[keep_final], scores_t[keep_final], classes_t[keep_final] 
                        inf_results = torch.stack(
                            (boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], 
                                scores, classes
                            )
                        )

                        results = Results(
                            orig_img=frames_images[f_id],
                            path= f"image_{f_id}.jpg", 
                            names=self.converter.class_names, 
                            boxes=inf_results.T, 
                            speed = {}, 
                            probs = classes 
                        )

                        detections = sv.Detections.from_ultralytics(results)
                        if self.tracker_choice == 'byte_tracker' and self.tracker is not None: 
                            detections = self.tracker.update_with_detections(detections) 
                        detections = detections[detections.tracker_id != -1] 

                        del results 
                        del inf_results 

                        results = self.build_results_from_detections(detections, frames_images[f_id], f_id)
                        self.results.append(results)
                        
                        f_inflight.discard(f_id) 
                        frames_images.pop(f_id,None)
            
                cur_host, cur_metas, n0, nxt_host, nxt_metas, n1 = nxt_host, nxt_metas, n1, cur_host, cur_metas, n0 
                self.run_callbacks("on_predict_postprocess_end") 

                if self.logic_module is not None and self.logic_module["DAV2"] is not None: 
                    fps = self.dataset.fps if self.dataset.mode == "video" else 30 
                    dav2_dir = os.getcwd() + '/assets/dav2_detections/'

                    if not os.path.exists(dav2_dir): 
                        os.makedirs(dav2_dir)

                    self.logic_module["DAV2"].detect(
                            queue=tbuf[self.seen:self.seen + len(self.results)], 
                            fps=fps, 
                            save_path=dav2_dir 
                    )

                n = len(self.results)
                for i in range(n): 
                    if self.seen == len(self.batch[1]): 
                        self.seen = 0

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
                        with StepContext(name="Write Results", catch=(RuntimeError, ), verbose=True):
                            self.batch[2][self.seen] += self.write_results(
                                    i = i,
                                    p = Path(self.batch[0][self.seen]),
                                    im = tbuf[:n0] ,
                                    original_images=self.batch[1],
                                    s = self.batch[2]
                            ) 

                        if producer_flag is not None : 
                           producer_flag.value = True 

                        if self.proc_image is not None and queue is not None: 
                            queue.put(self.proc_image) 
                        elif self.proc_image is None and queue is not None: 
                            queue.put(None) 

                    
                    with StepContext(name="Crop Objects to Image", catch=(RuntimeError,)):
                        self.capture_object_boxes(
                                image=self.batch[1][self.seen],
                                results=self.results[i],
                                cropped_dirname=self.cropped_image_dirname,
                                save=self.args.save
                        )
                    
                    self.seen += 1 
                
                self.results.clear()
                # if self.args.verbose: 
                #     logger.info("\n".join(self.batch[2])) 

                self.run_callbacks("on_predict_batch_end") 

                # yield from self.results 

        for v in self.vid_writer.values(): 
            if isinstance(v, cv2.VideoWriter): 
                v.release() 


        if self.args.verbose and self.seen: 
            t = tuple(x.t / self.seen * 1e3 for x in profilers) 
            logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *tbuf.shape[2:])}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")





























