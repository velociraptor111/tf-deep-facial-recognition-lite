"""
  Script to run face detection and recognition, using SSD Multibox Detector,MTCNN and Facenet for recognition.
  Written by: Peter Tanugraha
"""
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

from src.utils import *

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_custom.pb'
TARGET_ROOT_TEMP_DIR = './temp_roi'
FINAL_DETECTION_PATH = './final_detection'
FACENET_MODEL_PATH = './facenet/models/facenet/20180402-114759/20180402-114759.pb'
CLASSIFIER_PATH = './facenet/models/selfies_classifier_v2.pkl'

NUM_CLASSES = 2
CROP_SSD_PERCENTAGE = 0.3
IMAGE_SIZE = 160
FACENET_PREDICTION_BATCH_SIZE = 90
MAX_FRAME_COUNT = 1490

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
      
      cap = cv2.VideoCapture(0)

      while MAX_FRAME_COUNT:
        MAX_FRAME_COUNT -= 1
        ret, image = cap.read()
        if ret == 0:
            break
        initial_inference_start_time = time.time()
        if not os.path.isdir(TARGET_ROOT_TEMP_DIR):
          os.makedirs(TARGET_ROOT_TEMP_DIR)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB to pass on SSD detector
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

        bbox_dict = {}
        paths = []
        ids = []
        for detection_id,cur_det in enumerate(dets):
          boxes = cur_det[:4]
          (ymin, xmin, ymax, xmax) = (boxes[0] * im_height, boxes[1] * im_width,
                                        boxes[2] * im_height, boxes[3] * im_width)
          bbox = (xmin, xmax, ymin, ymax)
          new_xmin,new_xmax,new_ymin,new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax, CROP_SSD_PERCENTAGE, im_width, im_height)
          roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
          faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb,pnet, rnet, onet, image_size=IMAGE_SIZE)

          if len(faces_roi) != 0: # This is either a face or not a face
            faces_roi = faces_roi[0]
            faces_roi = faces_roi[:,:,::-1] #Convert from RGB to BGR to be compatible with cv2 image write
            cur_path = os.path.join(TARGET_ROOT_TEMP_DIR,'faces_roi_'+str(detection_id)+'.jpg')
            paths.append(cur_path)
            ids.append(detection_id)
            cv2.imwrite(cur_path,faces_roi)
            bbox_dict[detection_id] = bbox

        nrof_images = len(bbox_dict)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / FACENET_PREDICTION_BATCH_SIZE))
        embedding_size = embeddings.get_shape()[1]
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
          start_index = i * FACENET_PREDICTION_BATCH_SIZE
          end_index = min((i + 1) * FACENET_PREDICTION_BATCH_SIZE, nrof_images)
          paths_batch = paths[start_index:end_index] # Pass in several different paths
          images = facenet.load_data(paths_batch, False, False, IMAGE_SIZE)
          feed_dict = {images_placeholder: images, phase_train_placeholder: False}
          function_timer_start = time.time()
          emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
          function_timer = time.time() - function_timer_start
          print('Calculating image embedding cost: {}'.format(function_timer))

        shutil.rmtree(TARGET_ROOT_TEMP_DIR)
        ### Loading the SVM classifier ###
        with open(CLASSIFIER_PATH, 'rb') as infile:
          (model, class_names) = pickle.load(infile)

        if emb_array.shape[0] != 0:
            function_timer_start = time.time()
            predictions = model.predict_proba(emb_array)
            function_timer = time.time() - function_timer_start
            print('Predicting using SVM cost: {}'.format(function_timer))
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            elapsed_inference_time = time.time() - initial_inference_start_time
            print('Total inference time cost: {}'.format(elapsed_inference_time))

            print_recognition_output(best_class_indices, class_names, best_class_probabilities,
                                     recognition_threshold=0.7)
            draw_detection_box(image,ids,bbox_dict,class_names,best_class_indices,best_class_probabilities)
            cv2.imshow('video_view', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break




