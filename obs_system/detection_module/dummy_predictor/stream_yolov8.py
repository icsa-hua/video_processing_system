

from obs_system.detection_module.interface.streaming import YOLOStreamer
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imgsz
from ultralytics.data import load_inference_source
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.data.augment import classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path 
import torch 
import numpy as np
import cv2 
import re



class Yolov8Streamer(YOLOStreamer):

    # Ultralytics YOLO 🚀, AGPL-3.0 license
    """
    Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

    Usage - sources:
        $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                    img.jpg                         # image
                                                    vid.mp4                         # video
                                                    screen                          # screenshot
                                                    path/                           # directory
                                                    list.txt                        # list of images
                                                    list.streams                    # list of streams
                                                    'path/*.jpg'                    # glob
                                                    'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

    Usage - formats:
        $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                                yolov8n.torchscript        # TorchScript
                                yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                                yolov8n_openvino_model     # OpenVINO
                                yolov8n.engine             # TensorRT
                                yolov8n.mlpackage          # CoreML (macOS-only)
                                yolov8n_saved_model        # TensorFlow SavedModel
                                yolov8n.pb                 # TensorFlow GraphDef
                                yolov8n.tflite             # TensorFlow Lite
                                yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                yolov8n_paddle_model       # PaddlePaddle
                                yolov8n_ncnn_model         # NCNN
    """


    def __init__(self, cfg, overrides, _callbacks)->None: 
        super().__init__(cfg, overrides, _callbacks)


    def warmup(self, imgsz): 
        pass 


    def from_numpy(self, x): 
        pass 


    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one


    def pre_transform(self, im): 
        return super().pre_transform(im)
    

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im


    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)


    def postprocess(self, preds): 
        return super().postprocess(preds)
    
     
    def predict_cli(self, source, model): 
        gen = self.stream_inference(source, model)
        for _ in gen: 
            pass


    def setup_source(self, source=""): 
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
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
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(YOLOStreamer.STREAM_WARNING)
        self.vid_writer = {}


    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
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

    def non_max_suppression(self, opt, detections, iou_thres):
        pass 
    
    def translate_data(self)->None:
        pass
   
    @smart_inference_mode()
    def stream_inference(self, source, model, *args, **kwargs):
        
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

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
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def write_results(self, i, p, im, s)->str:
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
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

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
    

    def save_predicted_images(self, save_path, frame):
        return super().save_predicted_images(save_path, frame)
    

    def show(self, p): 
        return super().show(p)
    

    def run_callbacks(self, event): 
        return super().run_callbacks(event)
    

    def add_callback(self, event, func):
        return super().add_callback(event, func)















