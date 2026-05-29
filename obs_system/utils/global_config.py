# Global configuration file for Thresholds and configuration variables. 

#---------- Detection Thresholds ------------
CONF_THR = 0.25 
NMS_IOU = 0.45 
CLASS_AGNOSTIC = True 


#---------- Image Tiling parameters ---------
TILE_SIZE = 640
TILE_OVERLAP = 0.05
TILE_NMS_IOU = 0.30
TILE_THR = 3

#--------- Defish Parameters --------- 
DEFISH_K = 0.35
DEFISH_CROP = 0.05 
DISTORTION_STRENGTH = -0.20
DEFISH_ALPHA = 0.5
DEFISH_BETA = 10

#--------- Motion Gating ----------
TRIALS = 10 
HISTORY = 300 
VARTHRESHOLD = 32
THR_RATIO = 0.2 
K_CONSECUTIVE = 3 
HOLD_FRAMES = 10
MIN_OBJ_AREA = 0.003
EMPTY_IMAGE_PATH = "samples/highway_rescaled.png" 

#--------- Frame Batching ---------
BATCH_SIZE = 16
WARM_UP_SESSIONS = 8
FIXED_WIDTH = 1245
FIXED_HEIGHT = 1088


#-------- ROI COORDINATE TO CROP INITIAL IMAGE --------
ROI_X1 = 0 
ROI_Y1 = 639 
ROI_X2 = 430 
ROI_Y2 = 300
REGION_COLOR = (255, 42, 4) 


#-------- False Positives -------
MIN_WH = 16 
MULTIPLIER = 0.8


#-------- Object Vocabulary -----
VOCAB = ['car', 'bus', 'bike',
         'motorbike', 'motorcycle',
         'truck', 'cyclist', 'motorcyclist']

RED = (0, 0, 255)

#--------- Motion Quality Filtering (Step 1) ----------
# Fraction of total frame area; blobs smaller than this are discarded as noise
MIN_MOTION_COMPONENT_AREA_RATIO = 0.0002
# If the number of passing blobs exceeds this, assume vegetation scatter → suppress
MAX_MOTION_COMPONENTS = 25
# Morphological kernel size (pixels) used for the motion-gate cleanup pass
MOTION_MORPH_KERNEL = 5
# Minimum filtered-and-weighted motion score to pass the gate
MOTION_GATE_FILTERED_SCORE_THRESHOLD = 0.002

#--------- Unstable Motion Map (Step 2) ----------
ENABLE_UNSTABLE_MOTION_MAP = True
# Fraction of warmup frames in which a pixel must fire to be labelled "unstable"
UNSTABLE_MOTION_THRESHOLD = 0.40
# Contribution weight of an unstable pixel toward the motion score (0 = fully suppressed)
UNSTABLE_MOTION_SUPPRESSION_WEIGHT = 0.20

#--------- Drivable-Area Confidence Map (Step 3) ----------
ENABLE_DRIVABLE_CONFIDENCE_MAP = True
# Per-source contribution weights (must sum <= 1.0; remainder left to runtime learning)
DRIVABLE_STATIC_WEIGHT = 0.50
DRIVABLE_DETECTION_WEIGHT = 0.25
DRIVABLE_TRACK_WEIGHT = 0.15
DRIVABLE_UNSTABLE_NEGATIVE_WEIGHT = 0.10
# Pixels below this confidence are treated as "not reliably drivable"
DRIVABLE_CONFIDENCE_THRESHOLD = 0.25
