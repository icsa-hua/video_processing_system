from obs_system.logic_module.interface.event_extractor import EventExtractorInterface
from obs_system.utils.logger import get_logger 
import cv2 
import numpy as np 

from typing import List 

logger = get_logger("obs_system."+__name__)

class HomographicSetter(EventExtractorInterface): 

    def __init__(self, img_pts=None, grd_pts=None)->None: 

        # Image points (x,y). These represent the road plane (lane corners or similarly) 
        if img_pts is None: 
            img_pts = np.array([
                [98.0,367.0], 
                [401.0,357.0], 
                [1013.0,571.0], 
                [470.0,571.0], 
            ])

        # Ground-plane points (x, y) in meters. We can define the origin/scale. This is the shape of the ground plane 
        if grd_pts is None: 
            grd_pts = np.array([
                [0.0, 0.0],
                [7.4, 0.0], 
                [7.4, 8.0], 
                [0.0, 8.0]
            ])

        self.H, _ = cv2.findHomography(
            srcPoints=img_pts, 
            dstPoints=grd_pts, 
            method=0
        )


    # Returns the footpoint of the object we want. 
    def planar_image_conv(self, u, v): 
        X = self.H @ np.array([u, v, 1.0], dtype=np.float32) 
        result = (X[0]/X[2], X[1]/X[2])
        print(result)
        return result

    def detect(self, predictions, save=False)->list: 
        pass


    def birds_eye_view(self, img, m_w=10.0, m_h=12.0, px_per_m=50):

        W_out, H_out = int(m_w*px_per_m), int(m_h*px_per_m) 

        S = np.array(
         [[px_per_m, 0, 0], 
          [0,px_per_m, 0], 
          [0,0,1]], dtype=np.float32
        )

        H_img_to_brideye= S @ self.H 

        bird = cv2.warpPerspective(img, H_img_to_brideye, (W_out, H_out), flags=cv2.INTER_LINEAR) 

        # cv2.imwrite("Birdeye_VIEW.png", bird) 
        logger.debug("Birds eye view saved")
        return bird, H_img_to_brideye

    
    def draw_polyline(self, img, points, color=(0,225,0), closed=True, thickness=2):  
        p = np.int32(points).reshape(-1,1,2) 
        cv2.polylines(img, [p], closed, color, thickness)

    
    def visualize_homography(self, img_bgr, poly_pts, m_w=12.0, m_h=15.0, px_per_m=50): 

        W_out, H_out = int(m_w*px_per_m), int(m_h*px_per_m) 

        img = img_bgr.copy() 

        self.draw_polyline(
            img=img, 
            points=poly_pts
        )

        bird, H_img_to_brideye = self.birds_eye_view(img, m_w=m_w, m_h=m_h, px_per_m=px_per_m)

        poly = np.float32(poly_pts).reshape(-1,1,2) 
        poly_be = cv2.perspectiveTransform(poly, H_img_to_brideye).reshape(-1,2) 
        self.draw_polyline(img=bird, points=poly_be, color=(0,0,255))

        grid = bird.copy() 
        step = px_per_m 
        for x in range(0, W_out, step): 
            cv2.line(grid, (x,0), (x,H_out), (128,128,128),1)

        for y in range(0, H_out, step): 
            cv2.line(grid, (0,y), (W_out, y), (128,128,128), 1) 

        bird=cv2.addWeighted(grid, 0.35, bird, 0.65, 0.0) 

        vis = np.hstack([
            cv2.resize(img, (bird.shape[1], bird.shape[0])), 
            bird
        ])
        
        cv2.imshow("Left: source w/ ROI  |  Right: bird's-eye (red=projected ROI)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
         

        return bird, poly_be































