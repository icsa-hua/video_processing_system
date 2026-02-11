from multiprocessing.sharedctypes import Value
from obs_system.logic_module.interface.event_extractor import EventExtractorInterface 
from obs_system.utils.common import _empty_results

import torch
import numpy as np
import supervision as sv 

from typing import Any
from collections import defaultdict, deque
# from trackers import SORTTracker 
# from trackers.core.deepsort.tracker import DeepSORTTracker 
from ultralytics.engine.results import Results

class TrackerHandler(EventExtractorInterface): 

    def __init__(self, tracker_choice:str="byte_tracker", **kwargs): 

        self.color = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", 
            "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66"
            ])
        self.__tracker_choice = tracker_choice.lower() 
        self.__tracker = self._create_tracker(**kwargs)
        self.__tracker.reset()
        self.__history = defaultdict(lambda:deque(maxlen=30))
        self.__track_class = {}
        self.__box_anotator = sv.BoxAnnotator(color = self.color, color_lookup=sv.ColorLookup.TRACK)


    def _create_tracker(self, **kwargs): 
        if self.__tracker_choice == "sort": 
            # return SORTTracker() 
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "deepsort": 
            # return DeepSORTTracker() 
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == 'byte_tracker': 
            return sv.ByteTrack() 
        else: 
            raise ValueError(f"Unsupported tracker type: {self.__tracker_choice}")


    def _detect(self, predictions, orig_img, save=False)->list: 
        
        detections = sv.Detections.from_ultralytics(predictions)

        if self.__tracker_choice == "sort": 
            # return self.__tracker.update(detections)
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "deepsort": 
            # return self.__tracker.update(detections, orig_img) 
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "byte_tracker":
            return self.__tracker.update_with_detections(detections)
        else: 
            raise RuntimeError("Invalid tracker state")


    def detect(self, predictions, save:bool, orig_frame, f_id:int, class_names:list, speed:dict={}) -> Any: 

        detections = self._detect(predictions, save=save, orig_img=orig_frame)
        detections = detections[detections.tracker_id != -1] if isinstance(detections, sv.Detections) else [] 
        
        if len(detections) == 0: 
            return _empty_results(orig_image=orig_frame, frame_id=f_id, class_names=class_names)

        xyxy = torch.from_numpy(detections.xyxy).to(torch.float32)
        scores_t = torch.from_numpy(detections.confidence).to(torch.float32) 
        class_t = torch.from_numpy(detections.class_id).to(torch.int64) 
        ids_t = torch.from_numpy(detections.tracker_id).to(torch.int64) 
        
        results = torch.stack((xyxy[:,0],xyxy[:,1],xyxy[:,2],xyxy[:,3],ids_t.view(-1),scores_t.view(-1), class_t.view(-1)))

        return Results(orig_img=orig_frame, path=f"image_{f_id}.jpg", names=class_names, boxes=results.T, speed=speed)

    
    def update_tracker_history(self,results, logic_module:Any): 

        if results.boxes.id is None or self.__history is None: 
            return 

        boxes = results.boxes.xyxy.cpu().numpy() 
        tr_ids = results.boxes.id.int().cpu().numpy() 
        classes = results.boxes.cls.cpu().numpy()

        if boxes.size == 0: 
            return 
        bbox_center =np.column_stack((
                (boxes[:, 0] + boxes[:, 2]) * 0.5,
                (boxes[:, 1] + boxes[:, 3]) * 0.5,
         ))

        current_ids = set() 

        # Update tracking history
        for track_id,cls_i, centr in zip(tr_ids, classes, bbox_center): 
            current_ids.add(int(track_id)) 
            self.__track_class[int(track_id)] = int(cls_i)
            self.__history[int(track_id)].append(centr)

        points = {} 
        for track_id, cls, bcentr in zip(tr_ids,classes, bbox_center): 
            points[cls] = np.hstack(self.__history[track_id]).astype(np.int32).reshape((-1,1,2))
            if logic_module["ROI"] is not None: 
                logic_module["ROI"].count_regions(bbox=bcentr)

        #Remove track IDs from track history that were not detected in the current frame 
        lost_ids = set(self.__history.keys()) - current_ids 
        for tid in lost_ids: 
            self.__history.pop(tid, None)
            self.__track_class.pop(tid,None)
            points = {cls:pts for cls, pts in points.items() if cls not in lost_ids}

        return points 
                    






        



         
    




