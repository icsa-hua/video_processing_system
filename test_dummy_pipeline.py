from obs_system.camera_module.dummy_impl.video_consumer import DummyConsumer
from obs_system.detection_module.dummy_predictor.dummy_yolo import DummyPredictor
from obs_system.logic_module.dummy_logic.overlap_detection import BoundingBoxOverlapDetector
from obs_system.communication_module.dummy_com.message_transmitter import DummyMQTT
import cv2
import multiprocessing
import json



def main():
    parent_conn, child_conn = multiprocessing.Pipe()
    video_path = "samples/sample_video.mp4"
    camera_id = 0
    consumer = DummyConsumer(child_conn, camera_id, video_path)
    process = multiprocessing.Process(target=consumer.run)
    process.start()
    model_predictor = DummyPredictor("dummy_model_path")
    logic_module = BoundingBoxOverlapDetector()
    mqtt_topic = "test_topic"
    mqtt_host = "123.123.123.123"
    mqtt_com = DummyMQTT(mqtt_host)

    try:
        while True:
            if parent_conn.poll():
                try:
                    frame = parent_conn.recv()
                    if frame is not None:                         
                        cv2.imshow("frame", frame)
                        cv2.waitKey(1)

                        #Model prediction
                        predictions = model_predictor.predict(frame)
                        detect_frame = model_predictor.draw_boxes(frame.copy(), predictions)
                        cv2.imshow("Detection frame", detect_frame)
                        cv2.waitKey(1) 

                        #Logic module
                        result = logic_module.detect(predictions)
                        overlap_frame = logic_module.draw_overlaps(frame.copy(), result)
                        cv2.imshow("Overlap frame", overlap_frame)
                        cv2.waitKey(1)
             
                        #Communication module
                        mqtt_com.publish(mqtt_topic, str(result))

                    else:
                        print("Received None frame")
                except Exception as e:
                    print("Error while receiving/processing frame:", e)


    except KeyboardInterrupt:
        process.terminate()
        process.join()

if __name__ == "__main__":
    main()