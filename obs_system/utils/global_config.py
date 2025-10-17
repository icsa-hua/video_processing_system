# Global configuration file for Thresholds and configuration variables. 

#---------- Detection Thresholds ------------
CONF_THR = 0.25 
NMS_IOU = 0.45 
CLASS_AGNOSTIC = True 


#---------- Image Tiling parameters ---------
TILE_SIZE = 640 
TILE_OVERLAP = 0.25 


#--------- Defish Parameters --------- 
DEFISH_K = 0.35
DEFISH_CROP = 0.05 
DISTORTION_STRENGTH = -0.20
DEFISH_ALPHA = 0.5
DEFISH_BETA = 10


#--------- Motion Gating ----------
TRIALS = 10 
HISTORY = 500 
THR_RATIO = 0.2 
K_CONSECUTIVE = 3 
HOLD_FRAMES = 10
MIN_OBJ_AREA = 0.003


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
