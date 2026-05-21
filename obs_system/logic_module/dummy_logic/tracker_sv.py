from multiprocessing.sharedctypes import Value
from obs_system.detection_module.interface.detection_batch import FrameDetections
from obs_system.logic_module.interface.event_extractor import EventExtractorInterface 
from obs_system.utils.common import _empty_results

import torch
import numpy as np
import supervision as sv 

from typing import Any
from collections import defaultdict, deque
from trackers import SORTTracker 
#from trackers.core.deepsort.tracker import DeepSORTTracker 
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
        self.__history_len = 30
        self.__history = defaultdict(lambda:deque(maxlen=self.__history_len))
        self.__track_class = {}
        self.__box_anotator = sv.BoxAnnotator(color = self.color, color_lookup=sv.ColorLookup.TRACK)


    def _create_tracker(self, **kwargs): 
        if self.__tracker_choice == "sort": 
            return SORTTracker() 
            # raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "deepsort": 
            # return DeepSORTTracker() 
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == 'byte_tracker': 
            return sv.ByteTrack() 
        else: 
            raise ValueError(f"Unsupported tracker type: {self.__tracker_choice}")


    def _detect(self, predictions, orig_img, save=False)->list: 
        
        detections = sv.Detections.from_ultralytics(predictions)
        return self._update_with_detections(detections)

    def _detections_from_components(self, boxes, scores, classes) -> sv.Detections:
        if torch.is_tensor(boxes):
            boxes = boxes.detach().cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        if torch.is_tensor(classes):
            classes = classes.detach().cpu().numpy()

        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(scores, dtype=np.float32),
            class_id=np.asarray(classes, dtype=np.int64),
        )

    def _update_with_detections(self, detections: sv.Detections):
        if len(detections) == 0:
            return detections

        if self.__tracker_choice == "sort": 
            return self.__tracker.update(detections)
            # raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "deepsort": 
            # return self.__tracker.update(detections, orig_img) 
            raise ValueError("Not supported Tracker Type")
        elif self.__tracker_choice == "byte_tracker":
            return self.__tracker.update_with_detections(detections)
        else: 
            raise RuntimeError("Invalid tracker state")

    def _results_from_detections(self, detections: sv.Detections, orig_frame, f_id: int, class_names: list, speed: dict = {}) -> Any:
        detections = detections[detections.tracker_id != -1] if isinstance(detections, sv.Detections) else [] 

        if len(detections) == 0: 
            return _empty_results(orig_image=orig_frame, frame_id=f_id, class_names=class_names)

        xyxy = torch.from_numpy(detections.xyxy).to(torch.float32)
        scores_t = torch.from_numpy(detections.confidence).to(torch.float32) 
        class_t = torch.from_numpy(detections.class_id).to(torch.int64) 
        ids_t = torch.from_numpy(detections.tracker_id).to(torch.int64) 
        
        results = torch.stack((xyxy[:,0],xyxy[:,1],xyxy[:,2],xyxy[:,3],ids_t.view(-1),scores_t.view(-1), class_t.view(-1)))
        tracked = Results(orig_img=orig_frame, path=f"image_{f_id}.jpg", names=class_names, boxes=results.T, speed=speed)
        tracked.sv_detections = detections
        return tracked


    def detect(self, predictions, save:bool, orig_frame, f_id:int, class_names:list, speed:dict={}) -> Any: 

        detections = self._detect(predictions, save=save, orig_img=orig_frame)
        return self._results_from_detections(detections, orig_frame=orig_frame, f_id=f_id, class_names=class_names, speed=speed)

    def detect_compact(self, frame: FrameDetections, class_names: list, speed: dict = {}) -> Any:
        if frame.is_empty:
            return _empty_results(orig_image=frame.orig_img, frame_id=frame.frame_id, class_names=class_names)

        detections = self._detections_from_components(frame.boxes, frame.scores, frame.classes)
        detections = self._update_with_detections(detections)
        return self._results_from_detections(
            detections,
            orig_frame=frame.orig_img,
            f_id=int(frame.frame_id) if isinstance(frame.frame_id, (int, np.integer)) else frame.frame_id,
            class_names=class_names,
            speed=speed,
        )

    
    def update_tracker_history(self,results, logic_module:Any, build_points: bool = True): 

        if results is None or results.boxes is None:
            return {}

        if results.boxes.id is None or self.__history is None: 
            return {}

        detections = getattr(results, "sv_detections", None)
        if isinstance(detections, sv.Detections):
            boxes = detections.xyxy
            tr_ids = detections.tracker_id
            classes = detections.class_id
        else:
            boxes = results.boxes.xyxy.cpu().numpy() 
            tr_ids = results.boxes.id.int().cpu().numpy() 
            classes = results.boxes.cls.cpu().numpy()

        if boxes.size == 0: 
            return {}
        bbox_center = np.column_stack((
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
        for track_id, cls, bcentr in zip(tr_ids, classes, bbox_center):
            if logic_module["ROI"] is not None: 
                logic_module["ROI"].count_regions(bbox=bcentr)
            if not build_points:
                continue
            track_path = np.asarray(self.__history[int(track_id)], dtype=np.float32)
            if track_path.size == 0:
                continue
            points[cls] = track_path.astype(np.int32).reshape((-1, 1, 2))

        #Remove track IDs from track history that were not detected in the current frame 
        lost_ids = set(self.__history.keys()) - current_ids 
        for tid in lost_ids: 
            self.__history.pop(tid, None)
            self.__track_class.pop(tid,None)
            points = {cls:pts for cls, pts in points.items() if cls not in lost_ids}

        return points if build_points else {}

    def set_history_persistence(self, history_len: int) -> None:
        history_len = max(5, int(history_len))
        if history_len == self.__history_len:
            return

        self.__history_len = history_len
        old_items = list(self.__history.items())
        self.__history = defaultdict(lambda: deque(maxlen=self.__history_len))
        for tid, track_hist in old_items:
            self.__history[tid].extend(track_hist)
                    






        



         
    
