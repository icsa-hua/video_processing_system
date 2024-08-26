"Standalone Program. Run directly with python3"
import os
import time
import platform 
import re 
import threading 
from pathlib import Path 
import glob 
import cv2  
import numpy as np
import torch 
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.build import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
# from soft_nms import py_cpu_softnms
from obs_system.detection_module.interface.soft_nms import py_cpu_softnms
from obs_system.detection_module.interface.streaming import YOLOStreamer
import gc
import psutil
# import pynvml



STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""

class Yolov5Streamer(YOLOStreamer): 

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None): 
        super().__init__(cfg, overrides, _callbacks)
        # self.args = get_cfg(cfg,overrides)
        # self.save_dir = get_save_dir(self.args)

        # self.process_memory = psutil.Process(os.getpid())
        # pynvml.nvmlInit()
        # self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # if self.args.conf is None: 
        #     self.args.conf = 0.25 
        # self.done_warmup = False 
        
        # if self.args.show: 
        #     self.args.show = check_imshow(warn=True)
        
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
                
        # self.model = None 
        # self.data = self.args.data
        # self.imgsz = None 
        # self.device = None 
        # self.dataset = None 
        # self.vid_writer = {} 
        # self.plotted_img = None
        # self.source_type = None 
        # self.seen = 0
        # self.windows = [] 
        # self.batch = None 
        # self.results = None
        # self.transforms = None 
        # self.callbacks = _callbacks or callbacks.get_default_callbacks() 
        # self.txt_path = None 
        # self._lock = threading.Lock() 
        # self.shape = None 
        # callbacks.add_integration_callbacks(self)


    def _warmup(self, imgsz=(1, 3, 640, 640)):
        warmup_types = self.model.pt
        if warmup_types and self.device != 'cpu':
            im = torch.empty(*imgsz, dtype=torch.float, device=self.device)
            y = self.model(im, augment=False,profile=False)
            
            if isinstance(y, (list,tuple)): 
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else: 
                return self.from_numpy(y)


    def from_numpy(self, x): 
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        self.stream = stream
        if stream: 
            return self.stream_inference(source, model, *args, **kwargs)
        else: 
            return list(self.stream_inference(source, model, *args, **kwargs))


    def pre_transform(self, im): 
        same_shapes =  len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgsz,auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]
    

    def preprocess(self, im): 
        
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor: 
            im = np.stack(self.pre_transform(im))
            
            im = im[..., ::-1].transpose((0,3,1,2))
            im = np.ascontiguousarray(im)
            self.shape = im.shape
        
            im = torch.from_numpy(im)
        
        im = im.to(self.device)
        im = im.float()
        if not_tensor:
            im /= 255

        return im 
    

    def inference(self, im, *args, **kwargs):
      
        with torch.no_grad(): 
            preds = self.model(im,augment=self.args.augment)

        return preds


    def postprocess(self, preds, img, orig_imgs):
        return preds 


    def predict_cli(self, source=None, model=None): 
        gen = self.stream_inference(source, model)

        for _ in gen: 
            pass #sourcery skip: remove-empty-nested-block noqa


    def setup_source(self,source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        self.transform = (getattr(self.model.model, "transforms", classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction))
                          if self.args.task == "classify" else None)

        self.dataset = load_inference_source(source=source, batch=self.args.batch,
                                             vid_stride=self.args.vid_stride,buffer=self.args.stream_buffer)
        
        self.source_type = self.dataset.source_type

        if not getattr(self,"stream", True ) and \
           (self.source_type.stream or self.source_type.screenshot
            or len(self.dataset)>1000 or any(getattr(self.dataset, "video_flag", [False]))): 
            LOGGER.warning(STREAM_WARNING)


    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        if self.args.verbose:
            LOGGER.info("WARNING")

        if not self.model: 
            self.setup_model(model)

        init_time = time.time()
        with self._lock: 
            self.setup_source(source if source is not None else self.args.source)
            
            if self.args.save or self.args.save_txt: 
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            
            if not self.done_warmup:
                self._warmup(imgsz=(1 if self.model.pt else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None 
            profilers = [ops.Profile(device=self.device) for _ in range(3)]

            self.run_callbacks("on_predict_start")

            for self.batch in self.dataset: 
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                
                #Preprocess
                with profilers[0]: 
                    im = self.preprocess(im0s)

                #inference 
                with profilers[1]: 
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed: 
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds 
                        continue 


                #postprocess 
                with profilers[2]: 
                    self.results = self.postprocess(preds, im, im0s)
                
                self.run_callbacks("on_predict_postprocess_end")

                n = len(im0s)
                self.detections  = {i : [] for i in range(n)}

                for i in range(n): 
                    self.seen +=1 
                    self.results[i].speed = {
                        "preprocess":profilers[0].dt * 1e3 /n, 
                        "inference": profilers[1].dt *1e3 / n, 
                        "postprocess":profilers[2].dt * 1e3 /n
                    }

                    """
                    Here is the post processing where NMS takes place + creating the new frame with detections. 
                    This creates the bottleneck for the inference.
                    """
                    # if self.args.verbose or self.args.save or self.args.save_txt or self.args.show: 
                        # self.translate_results()
                        # self.draw_yolo_boxes(im, self.detections[i], save_path='')
                
                if self.args.verbose: 
                    LOGGER.info("\n".join(s))
                
                self.run_callbacks("on_predict_batch_end")
                yield from self.results 
        
        #Release assets 
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers) #speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop: 
            n1 = len(list(self.save_dir.glob("lables/*.txt"))) 
            s = f"\n{n1} label{'s'*(n1>1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold',self.save_dir)}{s}")
        
        self.run_callbacks("on_predict_end")
        

    def setup_model(self, model, verbose=True):
        
        import torch
        cuda_use = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_use else "cpu")
        if cuda_use :
            print("CUDA is available. GPU is working!")
        else:
            print("CUDA is not available. GPU is not working.")
    
        #Load Model 
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model, force_reload=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        self.model.conf = 0.25
        self.model.iou = 0.50
        self.model.multi_label = False
        self.model.max_det = 500

        # self.model = AutoBackend(
        #     weights=model or self.args.model, 
        #     device=select_device(self.args.device,verbose=verbose), 
        #     dnn=self.args.dnn, 
        #     data=self.args.data, 
        #     fp16=self.args.half, 
        #     # batch=self.args.batch, 
        #     fuse=True, 
        #     verbose=verbose
        # )
        
        # self.device =self.model.device 
        # self.args.half=self.model.fp16

        self.model.eval()


    def non_max_suppressions(self,predictions, iou_thres=0.5):
        import torchvision.ops as ops
        nms_detections = ops.nms(predictions[:,:4], predictions[:,4], iou_threshold=iou_thres)            
        return nms_detections 


    def soft_nms_detections(self,predictions,method=2): 
        from keras import backend as K
        import tensorflow as tf
        array = predictions
        with tf.compat.v1.Session() as sess:
            index = py_cpu_softnms(array[:,:4], array[:,4], method=method)
            selected_predictions = sess.run(tf.gather(array, index))
        
        return selected_predictions


    def translate_results(self):  
        batch_size = self.results.shape[0]   
        confidence_threshold = 0.4
        nc = 80

        for frame_idx in range(batch_size): 
            output = self.results[frame_idx] #results for a single image. 
           
            high_conf_mask = output[:,4] > confidence_threshold
            if not high_conf_mask.any():
                continue

            batch_output = output[high_conf_mask]
            if batch_output.shape[0] == 0:
                continue    

            output_np = batch_output.cpu().numpy()
            boxes = output_np[:, :4]
            confidences = output_np[:, 4]
            class_probs = output_np[:, 5:5 + nc]

            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0] 
            boxes[:, 3] += boxes[:, 1] 

            class_ids = np.argmax(class_probs, axis=1)
            class_confs = class_probs[np.arange(batch_output.shape[0]), class_ids]
            final_scores = confidences * class_confs


            rectBoxes = np.stack((boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], final_scores, class_ids), axis=-1)
            box_tensor = torch.from_numpy(rectBoxes)
            indices = self.non_max_suppressions(box_tensor)
            self.detections[frame_idx] = rectBoxes[indices]

            gc.collect()
 

    def draw_yolo_boxes(self,frame,predictions, save_path=''):

        frame = (frame.cpu().numpy() * 255).astype(np.uint8)

        
        if not predictions.any(): 
            return 
        
        for det in predictions: 
            if predictions.shape[0] == 0:
                continue
            [x1, y1, x2, y2 , score], class_id = map(int, det[:5]), det[5]

            cv2.rectangle(frame,(x1,y1), (x2,y2),(0, 255, 0), 2)
            cv2.putText(frame, f'{self.class_names[class_id]} {score:.2f}', (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        if self.dataset.mode in {"stream","video"}: 
            fps = self.dataset.fps if self.dataset.mode =="video" else 30
            frames_path = f'{save_path.split(".",1)[0]}_frames/'

            if save_path not in self.vid_writer: 
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                
                suffix, fourcc = {".mp4","avc1"} if MACOS else (".avi","WMV2") if WINDOWS else (".avi", "MJPG")
                save_path = 'twitch'

                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc), 
                    fps=fps, 
                    frameSize=(frame.shape[1], frame.shape[0])   
                )
                
            self.vid_writer[save_path].write(frame)
            if self.args.save_frames:
                frame_number = len(list(Path(frames_path).glob("*.jpg"))) + 1
                cv2.imwrite(f"{frames_path}{frame_number}.jpg", frame)

        else:
            cv2.imwrite(save_path, frame)


    def write_results(self, i, p, im, s):
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor: 
            string += f"{i}:"
            frame = self.dataset.count 
        else: 
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None 
        
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode =="image" else f"_{frame}"))
        string += "%gx%g " % im.shape[2:]
        result = self.results[i]

        result.save_dir = self.save_dir.__str__()
        string += f"{result.speed['inference']:.1f}ms"


        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width, 
                boxes=self.args.show_boxes, 
                conf=self.args.show_conf, 
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i]
            )
        
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
        im = self.plotted_img 

        if self.dataset.mode in {"strean","video"}: 
            fps = self.dataset.fps if self.dataset.mode =="video" else 30
            frames_path = f'{save_path.split(".",1)[0]}_frames/'
            if save_path not in self.vid_writer: 
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = {".mp4","avc1"} if MACOS else (".avi","WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc), 
                    fps=fps, 
                    frameSize=(im.shape[1], im.shape[0])   
                )
                
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im) 

        else:
            cv2.imwrite(save_path, im)
    

    def show(self,p=""):
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows: 
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])
        cv2.imshow(p,im)
        cv2.waitKey(300 if self.dataset.mode == 'image' else 1)


    def run_callbacks(self, event: str): 
        for callback in self.callbacks.get(event, []): 
            callback(self)


    def add_callback(self, event: str, func): 
        self.callbacks[event].append(func)


# def get_video_path(start_path,video_name,  file_extension=".mp4"): 
    
#     for root, dirs, files in os.walk(start_path): 
#         for file in files: 
#             if video_name in file:
#                 return os.path.join(root, file)
       
#     return None 


# def find_directory(directory_name, search_path): 
#     result = ""
#     for root, dirs, files in os.walk(search_path):
#         if directory_name in dirs:
#             result = os.path.join(root, directory_name)
#             break
#     return result


# def main(): 

#     main_path_file = "Obstacle_Recognition_Edge_Ai-main"
#     main_path = find_directory(main_path_file, "/home/icsa_jim/Jupyter Notebook")
#     video_name = "10_DrivingWith.mp4"
#     video_path = get_video_path(main_path,video_name)
#     e = StreamPredictor()
#     # source = cv2.VideoCapture(video_path)
#     source = video_path
#     model_path = get_video_path(main_path, "yolov5s.pt")

#     e.predict_cli(source=source, model=model_path)


# if __name__ == "__main__":
#     main()











