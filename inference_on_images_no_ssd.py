"""
  Script to run face detection and recognition, using MTCNN for face detection and Facenet for recognition.
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
import align.detect_face
from src.utils import *

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_custom.pb'
SOURCE_IM_PATH_ARRAY = ['./images/image_2.jpg', './images/image_3.jpg']
FINAL_DETECTION_PATH = './final_detection'
FACENET_MODEL_PATH = './facenet/models/facenet/20180402-114759/20180402-114759.pb'
CLASSIFIER_PATH = './facenet/models/selfies_classifier_v2.pkl'

NUM_CLASSES = 2
CROP_SSD_PERCENTAGE = 0.3
IMAGE_SIZE = 160
FACENET_PREDICTION_BATCH_SIZE = 90

if __name__ == "__main__":
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            ### Creating and Loading MTCNN ###
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            ### Creating and Loading the Facenet Graph ###
            images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)

            for image_id, SOURCE_IM_PATH in enumerate(SOURCE_IM_PATH_ARRAY):
                initial_inference_start_time = time.time()
                image = cv2.imread(SOURCE_IM_PATH)
                # image = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA) #Apparently when not resized its better ?
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                dets_faces, bboxes_faces = align_image_with_mtcnn_with_tf_graph(image_np, pnet, rnet, onet,
                                                                                image_size=IMAGE_SIZE)
                images_array = []
                for face_id, det in enumerate(dets_faces):
                    images_array.append(prewhiten(det))

                nrof_images = len(images_array)
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

                for i, bbox_face in enumerate(bboxes_faces):
                    cv2.rectangle(image, (bbox_face[0], bbox_face[1]), (bbox_face[2], bbox_face[3]), (0, 0, 255), 3)

                    if best_class_probabilities[i] > 0.7:
                        cv2.putText(image, class_names[best_class_indices[i]], (int(bbox_face[0]), int(bbox_face[1]) + 10),
                                    0, 0.6, (0, 255, 0))
                    else:
                        cv2.putText(image, 'Unknown Face', (int(bbox_face[0]), int(bbox_face[1]) + 10), 0, 0.6, (0, 255, 0))

                cv2.imwrite(os.path.join(FINAL_DETECTION_PATH, 'final_detection_no_ssd_' + str(image_id) + '.jpg'), image)

                print_recognition_output(best_class_indices, class_names, best_class_probabilities,
                                         recognition_threshold=0.7)





