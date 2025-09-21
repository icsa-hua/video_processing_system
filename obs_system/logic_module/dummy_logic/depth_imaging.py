from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from ultralytics.utils import DEFAULT_CFG,LOGGER, MACOS, WINDOWS,callbacks,ops
import numpy as np
import cv2
import torch
import os  
import matplotlib
from pathlib import Path 



class DepthImageProcessor(EventExtractorInterface):
    def __init__(self, encoder='vits', pred_only=True, margin_width=50, grayscale=False): 
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        parent_path = os.getcwd()
        depth_anything_path = os.path.join(parent_path, 'DepthAnythingV2')

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'{depth_anything_path}/checkpoints/depth_anything_v2_{encoder}.pth'))
        self.model = self.model.to(self.DEVICE).eval()
        self.precision = "fp16"
        if self.precision == "fp16": 
            

            self.model = self.model.half()
        else: 
            self.model = self.model.float()

        self.pred_only = pred_only 
        self.margin_width = margin_width
        self.cmap = matplotlib.colormaps.get('Spectral_r')
        self.grayscale = grayscale
        self.vid_writer = {}
        

    def detect(self, queue, fps, save_path ):
        frame_width, frame_height = queue[0].shape[1], queue[0].shape[0]
        if save_path not in self.vid_writer:
            suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")

            self.vid_writer[save_path] = cv2.VideoWriter(
                filename=str(Path(save_path).with_suffix(suffix)),
                fourcc=cv2.VideoWriter_fourcc(*fourcc),
                fps=fps,  # integer required, floats produce error in MP4 codec
                frameSize=(frame_width, frame_height),  # (width, height)
            )

        aspectRatio = frame_width / frame_height
        depth_height  = 518 #fixed 
        depth_width = round(depth_height * aspectRatio / 14) * 14
        depth_width = (depth_width // 14) * 14

        for frame in queue: 
          
            depth = self.model.infer_image(frame, 518,precision=self.precision, depthHeight=depth_height, depthWidth=depth_width)
            depth = (depth-depth.min()) / (depth.max()-depth.min()) * 255.0 
            depth = depth.cpu().numpy().astype(np.uint8)
            
            if self.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else: 
                depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if self.pred_only:
                self.vid_writer[save_path].write(depth)
            else:
                split_region = np.ones((frame_height, self.margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([frame, split_region, depth])
                
                self.vid_writer[save_path].write(combined_frame)

        
    def deallocate_resource(self): 
        if self.vid_writer is not None:
            self.vid_writer.release()
    

    
