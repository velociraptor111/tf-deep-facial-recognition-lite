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

import os
import time
import shutil
import math
import pickle
import configparser
import json
import numpy as np
import tensorflow as tf
import cv2
import _init_paths
from src.align_image_mtcnn import align_image_with_mtcnn_with_tf_graph
import facenet

from src.utils import *

'''
  Getting all the necessary config variables
'''
config = configparser.ConfigParser()
config.read('config.ini')

PATH_TO_CKPT = config.get("DEFAULT","PATH_TO_SSD_CKPT")
SOURCE_IM_PATH_DIR = os.path.join(os.getcwd(),config.get("DEFAULT","SOURCE_IM_PATH_DIRECTORY"))
SOURCE_IM_PATH_ARRAY = [os.path.join(SOURCE_IM_PATH_DIR,path) for
                                path in config.get("DEFAULT","SOURCE_IM_NAMES").split(',')]
FINAL_DETECTION_PATH = config.get("DEFAULT","PATH_TO_FINAL_DETECTION_DIRECTORY")
FACENET_MODEL_PATH = config.get("DEFAULT","PATH_TO_FACENET_MODEL")
CLASSIFIER_PATH = config.get("DEFAULT","PATH_TO_SVM_EMBEDDINGS_CLASSIFIER")

CROP_SSD_PERCENTAGE = float(config.get("DEFAULT","CROP_SSD_PERCENTAGE"))
IMAGE_SIZE = int(config.get("DEFAULT","IMAGE_SIZE"))
FACENET_PREDICTION_BATCH_SIZE = int(config.get("DEFAULT","FACENET_PREDICTION_BATCH_SIZE"))

if __name__ == "__main__":

  with tf.Graph().as_default():

    ### Creating and Loading the Single Shot Detector ###
    image_tensor, boxes_tensor, scores_tensor, \
                    classes_tensor, num_detections_tensor = load_tf_ssd_detection_graph(PATH_TO_CKPT)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():

      ### Creating and Loading MTCNN ###
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

      ### Creating and Loading the Facenet Graph ###
      images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)

      for image_id,SOURCE_IM_PATH in enumerate(SOURCE_IM_PATH_ARRAY):
        initial_inference_start_time = time.time()

        image = cv2.imread(SOURCE_IM_PATH)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image to RGB to pass on SSD detector
        image_np_expanded = np.expand_dims(image_np, axis=0)

        start_time_ssd_detection = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
                                                            [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                                                            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time_ssd_detection
        print('SSD inference time cost: {}'.format(elapsed_time))
        dets = post_process_ssd_predictions(boxes,scores,classes)
        im_height = image.shape[0]
        im_width = image.shape[1]

        # For each of the detection boxes in dets, need to pass it to Facenet after using the facenet load data
        ids = []
        bbox_dict = {}
        images_array = []

        for detection_id,cur_det in enumerate(dets):
          boxes = cur_det[:4]
          (ymin, xmin, ymax, xmax) = (boxes[0] * im_height, boxes[1] * im_width,
                                        boxes[2] * im_height, boxes[3] * im_width)
          bbox = (xmin, xmax, ymin, ymax)
          bbox_dict[detection_id] = bbox

          new_xmin,new_xmax,new_ymin,new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax, CROP_SSD_PERCENTAGE, im_width, im_height)

          roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
          faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb,pnet, rnet, onet, image_size=IMAGE_SIZE)

          if len(faces_roi) != 0: # This is either a face or not a face
            faces_roi = faces_roi[0]
            images_array.append(prewhiten(faces_roi))
            ids.append(detection_id)
            bbox_dict[detection_id] = bbox

        nrof_images = len(bbox_dict)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / FACENET_PREDICTION_BATCH_SIZE))

        emb_array = get_face_embeddings(sess, embeddings, images_placeholder, phase_train_placeholder,
                                        nrof_images, nrof_batches_per_epoch, FACENET_PREDICTION_BATCH_SIZE,
                                        images_array)

        ### Loading the SVM classifier ###
        with open(CLASSIFIER_PATH, 'rb') as infile:
          (model, class_names) = pickle.load(infile)
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        elapsed_inference_time = time.time() - initial_inference_start_time
        print('Total inference time cost: {}'.format(elapsed_inference_time))

        print_recognition_output(best_class_indices,class_names,best_class_probabilities,recognition_threshold=0.1)
        draw_detection_box(image,ids,bbox_dict,class_names,best_class_indices,best_class_probabilities,threshold = 0.4)

        print("Saving the final detection images to ",
                                      os.path.join(FINAL_DETECTION_PATH,'final_detection_'+str(image_id)+'.jpg'))

        cv2.imwrite(os.path.join(FINAL_DETECTION_PATH,'final_detection_'+str(image_id)+'.jpg'),image)





