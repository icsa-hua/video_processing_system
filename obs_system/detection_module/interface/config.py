#Config model 
from obs_system.Mask_RCNN.mrcnn.config import Config
from obs_system.Mask_RCNN.samples.coco.coco import CocoConfig


class Camera360Config(CocoConfig):
    NAME = 'camera360'
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1 
    NUM_CLASSES = 1 + 80 # Background + 80 classes