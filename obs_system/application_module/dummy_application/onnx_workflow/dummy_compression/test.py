"""
Test File, to not be used in the final application.
"""


from onnxruntime.quantization import QuantType,quantize_dynamic, CalibrationDataReader
import onnxruntime 
import onnx 
import cv2
from PIL import Image
import numpy as np
import os
import pdb


def format_yolov5(frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

class yolov5_cal_data_reader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = cv2.imread(image_filepath)
        pillow_img = format_yolov5(pillow_img)
        nchw_data = cv2.dnn.blobFromImage(pillow_img, 1 / 255.0, (height,width), swapRB=True)
        unconcatenated_batch_data.append(nchw_data)
        
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    
           
    return batch_data

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def xywh2xyxy(x):
#     # Convert (center x, center y, width, height) to (x1, y1, x2, y2)
#     y = np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

def postprocess_onnx_output(output, image_shape, conf_threshold=0.5, iou_threshold=0.4):
    # Extract the predictions
    predictions = output[0]  # [1, 25200, 85]
    
    # Apply sigmoid to object confidence and class scores
    # predictions[..., 4:] = sigmoid(predictions[..., 4:])

    # Filter out the predictions with low object confidence
    mask = predictions[..., 4] >= conf_threshold
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []

    # Convert bounding boxes from (center x, center y, width, height) to (x1, y1, x2, y2)
    print(predictions)
    pdb.set_trace()
    # boxes = xywh2xyxy(predictions[:, :4])
    boxes = predictions[:, :4]  
    # Scale boxes to the original image dimensions
    boxes[:, [0, 2]] *= image_shape[1]  # scale x coordinates
    boxes[:, [1, 3]] *= image_shape[0]  # scale y coordinates

    # Extract the confidence scores and class IDs
    confidences = predictions[:, 4]
    class_probs = predictions[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    
    final_boxes = []
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            final_boxes.append((boxes[i], confidences[i], class_ids[i]))

    return final_boxes

def draw_boxes(image, boxes, class_names):
    for (box, confidence, class_id) in boxes:
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[class_id]}: {confidence:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

model_input = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/yolov5s.onnx"
output_model_path = "yolov5s_quantized.onnx"

dr = yolov5_cal_data_reader("/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/converted_mp4/generated_patches",model_input)

quantize_dynamic(
            model_input=model_input,
            model_output=output_model_path,
            # calibration_data_reader=dr,
            # activation_type=QuantType.QUInt8,  
            weight_type=QuantType.QUInt8, 
            per_channel=False,
            reduce_range=True
            # nodes_to_exclude=['/model.24/Mul_1','/model.24/Mul_3','/model.24/Concat',
            #                 '/model.24/Mul_5','/model.24/Mul_7','/model.24/Concat_1',
            #                 '/model.24/Mul_9','/model.24/Mul_11','/model.24/Concat_2',
            #                 '/model.24/Reshape_1','/model.24/Reshape_3','/model.24/Reshape_5',
            #                 '/model.24/Concat_3']
        )

import time
from onnxconverter_common import float16
# model = onnx.load(output_model_path)
# onnx.checker.check_model(model)
model = onnx.load(model_input)
model = float16.convert_float_to_float16(model)
# pdb.set_trace()
image_path = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/converted_mp4/generated_patches/patch_2_2024-06-10_12-20-39.png"
img = Image.open(image_path)
img = cv2.imread(image_path)
original_shape = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img_array = img/255.0

img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)

class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
session = onnxruntime.InferenceSession(model_input)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: img_array})[0]
pdb.set_trace() 
res = postprocess_onnx_output(output, original_shape)
output_image = cv2.imread(image_path)
output_image = draw_boxes(output_image, res, class_names)
output_image = output_image.astype(np.uint8)

cv2.imshow('Detected Objects', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(90)















# image_path = "/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/converted_mp4/generated_patches/patch_2_2024-06-10_12-20-39.png"
# img = Image.open(image_path)

# img = cv2.imread(image_path)
# original_shape = img.shape
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img.astype(np.float32)
# img_array = img/255.0

# img_array = np.transpose(img_array, (2, 0, 1))
# img_array = np.expand_dims(img_array, axis=0)



# print(img.shape)
# img = np.expand_dims(img, axis=0)
# pr = np.transpose(img)
# class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# print(pr.shape)
# session = onnxruntime.InferenceSession(model_path)
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# output = session.run([output_name], {input_name: img_array})[0]

# res = postprocess_onnx_output(output, original_shape)
# output_image = cv2.imread(image_path)
# output_image = draw_boxes(output_image, res, class_names)
# output_image = output_image.astype(np.uint8)

# # Display the image with bounding boxes
# cv2.imshow('Detected Objects', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()