from obs_system.compressed.interface.tensor_yolo import TensorRTYOLO
from obs_system.detection_module.interface.streaming import YOLOStreamer
from obs_system.utils.tiles import *
from obs_system.utils.appraisal import StepContext
from obs_system.utils.logger import get_logger 

import os 
import gc
import pdb
import torch
import cv2
import numpy as np 
import supervision as sv

from typing import Any
from pathlib import Path 
from ultralytics import YOLO
from torch.profiler import profile, ProfilerActivity 
from trackers import SORTTracker 
from torchvision.ops import batched_nms, nms
from trackers.core.deepsort.tracker import DeepSORTTracker 
from collections import defaultdict, deque
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import colorstr 


logger = get_logger("obs_system."+__name__)


class TensorRTRTXStreamer(YOLOStreamer): 

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
        ))

        results = Results(
            orig_img=orig_image, 
            path=f"image{image}.jpg", 
            names=self.converter.class_names, 
            boxes=inf_results.T, 
            speed={}, 
            probs=class_t
        )

        return results



    def setup_model(self, model, verbose=True, opt='tracking'): 

        device = select_device(self.args.device, verbose=verbose) 
        model_path = 'obs_system/compressed/yolov8s.onnx'
        self.model = TensorRTYOLO(engine_path=model_path)
        [self.height, self.width] = self.model.input_height, self.model.input_width 

        self.device = device 

        if opt == 'tracking':
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

        logger.info(f"[checked] Model {model} successfully set up")


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


    @smart_inference_mode()
    def stream_inference(self, source, model, producer_flag, queue, *args, **kwargs): 

        if self.args.verbose: logger.info("") 

        self.source = source
        with self._lock: 
            self.setup_source(source if source is not None else self.args.source) # Create Dataset obj. 

            pending = {} 
            frames_images = {}
            max_f_inflight = 16 # Frames batch size 
            micro = 32 # Maximum number for tiles to exist. 
            f_inflight = set() 
            frames_queue = deque()
            tile_queue = deque() 


            def enqueue_frame(frame_id, img): 
                self.orig_height, self.orig_width = img.shape[:2] 
                tiles = split_image(img, frame_id=frame_id, tile_size=640, overlap=0.15)
            
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
                    self.model.warmup(micro=micro, warmup_sessions=4)
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
            host0 = np.empty((micro, 640, 640, 3), np.uint8) 
            host1 = np.empty_like(host0) 

            metas0, metas1 = [None]*micro, [None]*micro 
            tbuf = torch.empty((micro, 3, 640, 640), device = self.device, dtype=torch.float32)

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
                    tbuf[:n0].copy_(tb, non_blocking=True) 

                with profilers[1]:
                    if self.seen == 0: 
                        with profile(activities=activities) as prof:
                            i_boxes, i_scores, i_classes = self.model(tbuf[:n0]) 
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
                            mapped_boxes = np.empty((0,4), np.float32) 
                            score = np.empty((0,), np.float32) 
                            cls_ = np.empty((0,), np.int64) 

                        else: 
                            mapped_boxes = reconstruct_tiles(
                                boxes_xyxy=det, 
                                tx=meta['left_x'], 
                                ty=meta['top_y'], 
                                orig_H=self.orig_height, 
                                orig_W=self.orig_width, 
                                gain=meta['gain'], 
                                pad=(meta['pad_x'],meta['pad_y'])
                            )

                        s["parts"].append( (mapped_boxes, score, cls_) ) 

                        if len(s["seen"]) == s["need"]: 

                            if len(s["parts"]) == 0 or all(p[0].shape[0] == 0 for p in s["parts"]): 
                                frames_out[f_id] = (
                                    np.empty((0,4), np.float32), 
                                    np.empty((0,), np.float32), 
                                    np.empty((0,), np.int64)
                                ) 

                            else: 
                                frames_out[f_id] = (
                                    np.concatenate([p[0] for p in s["parts"]], 0),
                                    np.concatenate([p[1] for p in s["parts"]], 0), 
                                    np.concatenate([p[2] for p in s["parts"]], 0)
                                )

                            del pending[f_id] 
                        
                    del i_boxes
                    del i_classes
                    del i_scores
                
                    for f_id in (frames_out.keys()):

                        out_image = frames_images.get(f_id)
                        if out_image is None: continue
                        boxes, scores, classes = frames_out[f_id]
                        boxes_t = torch.from_numpy(boxes)
                        scores_t = torch.from_numpy(scores) 
                        classes_t = torch.from_numpy(classes) 

                        if boxes is None or len(boxes) == 0: 
                            cv2.imwrite(os.path.join(out_dir, f"{f_id}.jpg"), out_image)
                            continue

                        keep_pc = batched_nms(boxes_t, scores_t, classes_t.long(), iou_threshold=0.2)

                        if keep_pc.numel() == 0:
                            keep_final = keep_pc  # empty
                        else:
                            boxes_pc  = boxes_t[keep_pc]
                            scores_pc = scores_t[keep_pc]

                            keep_ca = nms(boxes_pc, scores_pc, iou_threshold=0.4)

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



def flatten_tiles(tiles_batch): 
    for fb in tiles_batch: 
        fid = fb[0][1]['frame_id']
        for t_idx, (tile, meta) in enumerate(fb): 
            yield tile, {"frame_id":fid, "t_idx":t_idx, **meta}


def reconstruct_tiles(boxes_xyxy, tx, ty, orig_H, orig_W, gain=1, pad=(0,0)) : 

    pw, ph = pad 
    boxes_xyxy[:, 0::2] -= pw 
    boxes_xyxy[:, 1::2] -= ph 
    boxes_xyxy /= gain 

    boxes_xyxy[:,0::2] += tx 
    boxes_xyxy[:,1::2] += ty 

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, orig_W - 1) 
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, orig_H - 1) 

    return boxes_xyxy 



def fill(host, metas, tile_queue, micro): 
    n = 0 
    while  n < micro and tile_queue: 
        tile, meta = tile_queue.popleft() 
        host[n][...] = tile 
        metas[n] = meta 
        n += 1 

    for i in range(n, micro): 
        metas[i] = None 

        
        
    return n, host, metas, tile_queue
























