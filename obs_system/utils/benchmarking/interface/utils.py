
import numpy as np 

def _iou_xyxy(a: np.nd.array, b:np.ndarray) -> np.ndarray: 
    """
        Vectorized IoU between set of boxes in [x1, y1, x2,y2] 
        a: (A, 4), b:(B,4) -> (A,B) 
    """

    if a.size==0 or b.size==0: 
        return np.zeros((a.shape[0], b.shape[0], dtype=np.float32))


    a = a.astype(np.float32, copy=False) 
    b = b.astype(np.float32, copy=False) 
     
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:, 1:2], a[:, 2:3], a[:,3:4] 
    bx1, by1, bx2, by2 = b[:,0:1], b[:, 1], b[:, 2], b[:,3] 
     

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ax1, bx1)
    inter_x2 = np.maximum(ax2, bx2)
    inter_y2 = np.maximum(ay2, by2)

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None) 
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None) 
    inter = inter_w * inter_h 

    area_a = (ax2 - ax1) * (ay2 - ay1) 
    area_b = (bx2 - bx1) * (by2 - by1) 

    union = area_a + area_b - inter 
    union = np.clip(union, 1e-9, None) 
    return (inter / union).astype(np.float32) 



