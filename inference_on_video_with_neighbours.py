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

import numpy as np
import tensorflow as tf
import cv2

import _init_paths
from src.align_image_mtcnn import align_image_with_mtcnn_with_tf_graph
import facenet

from align.detect_face import create_mtcnn
from src.align_image_mtcnn import align_image_with_mtcnn_with_tf_graph
from src.utils import load_tf_ssd_detection_graph,run_inference_for_single_image,post_process_ssd_predictions,load_tf_facenet_graph,crop_ssd_prediction,prewhiten,get_face_embeddings,print_recognition_output,draw_detection_box

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = config.get("DEFAULT","PATH_TO_SSD_CKPT")
FINAL_DETECTION_PATH = config.get("DEFAULT","PATH_TO_FINAL_DETECTION_DIRECTORY")
FACENET_MODEL_PATH = config.get("DEFAULT","PATH_TO_FACENET_MODEL")
CLASSIFIER_PATH = config.get("DEFAULT","PATH_TO_SVM_EMBEDDINGS_CLASSIFIER")

CROP_SSD_PERCENTAGE = float(config.get("DEFAULT","CROP_SSD_PERCENTAGE"))
IMAGE_SIZE = int(config.get("DEFAULT","IMAGE_SIZE"))
FACENET_PREDICTION_BATCH_SIZE = int(config.get("DEFAULT","FACENET_PREDICTION_BATCH_SIZE"))
MAX_FRAME_COUNT = int(config.get("DEFAULT","MAX_FRAME_COUNT"))

CLASSIFIER_PATH_SVM = '/Users/petertanugraha/Projects/facenet/svm_classifier_models/peter_classifier.pkl'
CLASSIFIER_PATH_KNN = '/Users/petertanugraha/Projects/facenet/svm_classifier_models/peter_classifier_k_nearest_neighbours_clf.pkl'

if __name__ == "__main__":

    with tf.Graph().as_default():

        ### Creating and Loading the Single Shot Detector ###
        image_tensor, tensor_dict = load_tf_ssd_detection_graph(PATH_TO_CKPT, input_graph=None)

        sess = tf.Session()
        with sess.as_default():
            ### Creating and Loading MTCNN ###
            pnet, rnet, onet = create_mtcnn(sess, None)
            ### Creating and Loading the Facenet Graph ###
            images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)

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
                output_dict = run_inference_for_single_image(sess, image_np, image_tensor, tensor_dict)
                elapsed_time = time.time() - start_time_ssd_detection
                dets = post_process_ssd_predictions(image_np, output_dict, threshold=0.25)
                print('SSD inference time cost: {}'.format(elapsed_time))

                bbox_dict = {}
                ids = []
                images_array = []

                for detection_id, cur_det in enumerate(dets):
                    boxes = cur_det[:4]
                    (ymin, xmin, ymax, xmax) = (boxes[0], boxes[1],
                                                boxes[2], boxes[3])
                    bbox = (xmin, xmax, ymin, ymax)
                    new_xmin, new_xmax, new_ymin, new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax,
                                                                                 None, image_np.shape[1],
                                                                                 image_np.shape[0],0.5,0.3)
                    roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
                    roi_cropped_rgb = cv2.resize(roi_cropped_rgb, (250, 250))
                    faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb, pnet, rnet, onet,
                                                                        image_size=IMAGE_SIZE)

                    if len(faces_roi) != 0:  # This is either a face or not a face
                        faces_roi = faces_roi[0]
                        images_array.append(prewhiten(faces_roi))
                        ids.append(detection_id)
                        bbox_dict[detection_id] = bbox

                nrof_images = len(bbox_dict)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / FACENET_PREDICTION_BATCH_SIZE))
                emb_array = get_face_embeddings(sess, embeddings, images_placeholder, phase_train_placeholder,
                                                nrof_images, nrof_batches_per_epoch, FACENET_PREDICTION_BATCH_SIZE,
                                                images_array)

                ### Loading the SVM Classifier ###
                with open(CLASSIFIER_PATH_SVM, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                ### Loading the KNN Classifier ###
                with open(CLASSIFIER_PATH_KNN, 'rb') as infile:
                    knn_model = pickle.load(infile)

                if emb_array.shape[0] != 0:
                    distances, indices = knn_model.kneighbors(emb_array)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    average_distance_array = np.mean(distances,axis=1)

                    print("Average distance is: ", average_distance_array)
                    for i,id in enumerate(ids):
                        bbox = bbox_dict[id]
                        cv2.rectangle(image_display, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])),
                                      (255, 0, 0), 2)

                        if average_distance_array[i] < 0.85:
                            cv2.putText(image_display, class_names[best_class_indices[i]], (int(bbox[0]), int(bbox[2]) + 10), 0,
                                        0.8, (0, 255, 0), thickness=2)
                        else:
                            cv2.putText(image_display, 'Unknown Face', (int(bbox[0]), int(bbox[2]) + 10), 0, 0.8,
                                        (0, 0, 255), thickness=2)

                cv2.imshow('full-face-detection-pipeline', image_display)
                if cv2.waitKey(1) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()





