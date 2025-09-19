from interface.compressed_yolo import CompressedYOLO 
from interface.convert_to_Results import ConverterResults 

import os 
import torch 
import supervision as sv 
import pdb 

from trackers import SORTTracker 
from trackers.core.deepsort.tracker import DeepSORTTracker
from ultralytics.engine.results import Results 



tracker_choice = 'sort'
converter = ConverterResults() 

parent_dir = os.getcwd() 

sample_path = f'{parent_dir}/samples/sample_video.mp4'
device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu') 

print('Running on : ', device) 

if tracker_choice == 'sort': 
    tracker = SORTTracker() 
else:
    tracker = DeepSORTTracker()

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", 
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66"
    ])

box_annotator = sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK) 

CONFIDENCE_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.4 
frame_samples = [] 

TARGET_VIDEO_PATH = f'{tracker_choice}_results.mp4'
 
model_path = 'compressed/yolov8s.onnx' 
model = CompressedYOLO(model_path) 
[height, width] = model.input_height, model.input_width

def callback(frame, i): 
    boxes, scores, class_ids = model(frame)
    pad_x, pad_y, scale = converter.calculate_padding(height, width, 640) 
    boxes, scores, class_ids = converter.data_to_tensor_filter(boxes=boxes, scores=scores, class_ids=class_ids)

    if len(boxes) != 0: 
        boxes = converter.scale_boxes(boxes, pad_x=pad_x, pad_y=pad_y, scale=scale) 
        results = torch.stack((boxes[:, 0], boxes[:,1], boxes[:,2],boxes[:,3], scores, class_ids),axis=-1) 

        results = Results(
            orig_img = frame, 
            path=sample_path,
            names=converter.class_names, 
            boxes=results, 
            speed={}, 
            probs=class_ids 
        )

        detections = sv.Detections.from_ultralytics(results) 
        if tracker_choice == 'sort': 
            detections = tracker.update(detections)
        else : 
            detections = tracker.update(detections, frame)

        detections = detections[detections.tracker_id != -1]
        annotated_image = box_annotator.annotate(frame, detections)

        if i % 30 == 0 and i != 0: 
            frame_samples.append(annotated_image)

        return annotated_image



tracker.reset() 
sv.process_video(
    source_path = sample_path, 
    target_path=TARGET_VIDEO_PATH, 
    callback=callback,
)

sv.plot_images_grid(images=frame_samples[:4], grid_size=(2,2)) 










