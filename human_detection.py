"""
  Script to run face detection and recognition, using SSD Multibox Detector,MTCNN and Facenet for recognition.
  Written by: Peter Tanugraha
"""

# MIT License
#
# Copyright (c) 2018 Peter Tanugraha
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import numpy as np
import tensorflow as tf
import cv2

from src.utils import load_tf_ssd_detection_graph,run_inference_for_single_image_through_ssd,post_process_ssd_predictions\
    ,load_tf_facenet_graph,crop_ssd_prediction,prewhiten,get_face_embeddings,print_recognition_output,draw_detection_box

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_PERSON_DETECTION = config.get("DEFAULT","PATH_TO_PERSON_DETECTION")

if __name__ == "__main__":

    with tf.Graph().as_default():

        ### Creating and Loading the Single Shot Detector ###
        image_tensor, tensor_dict = load_tf_ssd_detection_graph(PATH_TO_PERSON_DETECTION, input_graph=None)

        sess = tf.Session()
        with sess.as_default():
            cap = cv2.VideoCapture(0)

            if cap.isOpened() is False:
                print("Error opening video stream or file")

            while cap.isOpened():
                _, image = cap.read()
                image = image[..., ::-1, :]
                image_display = image.copy()

                initial_inference_start_time = time.time()
                # Both the SSD and Facenet also uses np.uint8 and RGB images for both!
                image_np = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.uint8)

                start_time_ssd_detection = time.time()
                output_dict = run_inference_for_single_image_through_ssd(sess, image_np, image_tensor, tensor_dict)
                elapsed_time = time.time() - start_time_ssd_detection
                # In coco dataset label map is a human being
                dets = post_process_ssd_predictions(image_np, output_dict, threshold=0.5,detection_classes = [1])
                print('SSD inference time cost: {}'.format(elapsed_time))

                for detection_id, cur_det in enumerate(dets):
                    boxes = cur_det[:4]
                    (ymin, xmin, ymax, xmax) = (boxes[0], boxes[1],
                                                boxes[2], boxes[3])
                    cv2.rectangle(image_display, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                  (255, 0, 0), 2)

                cv2.imshow('full-face-detection-pipeline', image_display)
                if cv2.waitKey(1) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()





