import cv2
import torch 
import numpy as np 
import torchvision.ops as operation

from ultralytics.engine.results import Results 

class ConverterResults: 

    def __init__(self):
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    def translate_data(self, frame_index, video_path, images, results, height, width):
        '''
        Translates the data from the results to the format required by the tracker.
        ''' 

        shape = len(results) 
        rectBoxes, class_ids = self.iteration_rows(results, frame_index, shape, height, width)

        if len(rectBoxes) == 0: 
            return Results(
                orig_img=np.ndarray(0,0), 
                boxes=[],
                path="", 
                names=[], 
                masks=[], 
                probs=[], 
                speed={},   
            ) 
        
        return Results(
            orig_img=images, 
            path=video_path, 
            names=self.class_names,
            boxes=rectBoxes,
            speed={},
            probs=class_ids
        )

        
    def iteration_rows(self, results, frame_index, shape, height, width, conf_thr=0.4): 
        boxes, class_ids, scores = [],[],[] 
        pad_x, pad_y, scale = self.calculate_padding(height,width,640) 

        for r in range(results.shape[-1]): 
            class_scores = self.class_scores_creation(results, frame_index, r, shape)
            (_, maxScore, _, (x, maxClassIndex)) = cv2.minMaxLoc(class_scores.cpu().numpy()) 
            
            if maxScore >= conf_thr:
               boxes.append(self.box_creation(results, frame_index, r, shape))
               class_ids.append(maxClassIndex)
               scores.append(maxScore)

        try: 
            boxes, scores, class_ids = self.data_to_tensor_filter(boxes=boxes, scores=scores, class_ids=class_ids)
            boxes = self.scale_boxes(boxes, pad_x=pad_x, pad_y=pad_y, scale=scale)
            return torch.stack((boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],scores, class_ids), axis=-1), class_ids

        except:
            boxes, scores,class_ids  = [],[],[]
            return torch.empty((0,6)), []


    def calculate_padding(self, original_height, original_width, target_dimension):
        
        # Calculate the scale needed to fit the width and height into the target dimension
        # Determine which scale to use (the one that fits the image entirely within the target box)
        scale_used = min(target_dimension / original_width, target_dimension / original_height)

        # Calculate the effective width and height after scaling
        # Calculate padding by subtracting the effective dimensions from the target dimension
        effective_width = original_width * scale_used
        effective_height = original_height * scale_used
        
        return target_dimension - effective_width, target_dimension - effective_height, scale_used
            

    def box_creation(self, results, i, r, shape): 
        
        if shape == 3: 
            return [results[i, 0, r] - results[i, 2, r]/2,
                    results[i, 1, r] - results[i, 3, r]/2,
                    results[i, 0, r] + results[i, 2, r],
                    results[i, 1, r] + results[i, 3, r]]
        
        return [results[0, r] - results[2, r]/2,
                results[1, r] - results[3, r]/2,
                results[0, r] + results[2, r],
                results[1, r] + results[3, r]]


    def scale_boxes(self, boxes, pad_x, pad_y, scale): 
       boxes[:,[0,2]] -= pad_x // 2
       boxes[:,[1,3]] -= pad_y // 2 
       boxes[:, :4] /= scale
       return boxes
    

    def class_scores_creation(self,results, i, r, shape): 
        if shape == 3:
            return results[i, 4:, r]
        return results[4:, r]
    
    
    def non_max_suppression(self,detections, scores, iou:float):
        if len(detections)==0:
            return[]
        return operation.nms(detections, scores, iou_threshold=iou)            
            

    def data_to_tensor_filter(self,boxes, scores, class_ids): 
        boxes = torch.FloatTensor(boxes)
        scores = torch.FloatTensor(scores)
        class_ids = torch.LongTensor(class_ids)
        return boxes, scores, class_ids
