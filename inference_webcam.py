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

if __name__ == "__main__":

  ## Load SSD Tensorflow Graph ###
  function_timer_start = time.time()
  detection_graph, image_tensor, boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor = load_tf_ssd_detection_graph(PATH_TO_CKPT)
  function_timer = time.time() - function_timer_start
  print('Loading in the SSD Model cost: {}'.format(function_timer))

  # ### Load Facenet Tensorflow Graph ###
  # function_timer_start = time.time()
  # facenet_graph, images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)
  # function_timer = time.time() - function_timer_start
  # print('Loading image embedding model cost: {}'.format(function_timer))
  #
  # ### Load MTCNN Tensorflow Graph ###
  # function_timer_start = time.time()
  # MTCNN_graph, pnet, rnet, onet = load_tf_mtcnn_graph()
  # function_timer = time.time() - function_timer_start
  # print('Loading image MTCNN model cost: {}'.format(function_timer))

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
  with sess.as_default():
    ### Creating and Loading MTCNN ###
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    ### Creating and Loading the Facenet Graph ###
    facenet.load_model(FACENET_MODEL_PATH)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    frame_num = 1490

    cap = cv2.VideoCapture(0)

    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break
      initial_inference_start_time = time.time()
      if not os.path.isdir(TARGET_ROOT_TEMP_DIR):
        os.makedirs(TARGET_ROOT_TEMP_DIR)

      # image = cv2.imread(SOURCE_IM_PATH)
      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image to RGB to pass on SSD detector
      image_np_expanded = np.expand_dims(image_np, axis=0)
      with tf.Session( graph = detection_graph ) as sess_detect:
        start_time_ssd_detection = time.time()
        (boxes, scores, classes, num_detections) = sess_detect.run(
                                                            [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                                                            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time_ssd_detection
        print('SSD inference time cost: {}'.format(elapsed_time))

      dets = post_process_ssd_predictions(boxes,scores,classes)

      im_height = image.shape[0]
      im_width = image.shape[1]

      bbox_dict = {}  #This dictionary will hold all the bounding box location associated with the detection_id per image

      # For each of the detection boxes in dets, need to pass it to Facenet after using the facenet load data
      for detection_id,cur_det in enumerate(dets):
        boxes = cur_det[:4]
        (ymin, xmin, ymax, xmax) = (boxes[0] * im_height, boxes[1] * im_width,
                                      boxes[2] * im_height, boxes[3] * im_width)
        bbox = (xmin, xmax, ymin, ymax)
        bbox_dict[detection_id] = bbox

        new_xmin,new_xmax,new_ymin,new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax, CROP_SSD_PERCENTAGE, im_width, im_height)

        roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
        faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb,pnet, rnet, onet, image_size=IMAGE_SIZE)
        assert(len(faces_roi) == 1) # Take into account that SSD has already only detected one face!
        faces_roi = faces_roi[0]
        faces_roi = faces_roi[:,:,::-1] #Convert from RGB to BGR to be compatible with cv2 image write
        cv2.imwrite(os.path.join(TARGET_ROOT_TEMP_DIR,'faces_roi_'+str(detection_id)+'.jpg'),faces_roi)

      nrof_images = len(dets)
      nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / FACENET_PREDICTION_BATCH_SIZE))
      embedding_size = embeddings.get_shape()[1]
      emb_array = np.zeros((nrof_images, embedding_size))

      paths = []
      ids = []
      for path in os.listdir(TARGET_ROOT_TEMP_DIR):
        if path.endswith('.jpg'):
          paths.append(os.path.join(TARGET_ROOT_TEMP_DIR,path))
          ids.append(int(path.split('_')[-1].split('.')[0]))

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
      function_timer_start = time.time()
      ### Loading the SVM classifier ###
      with open(CLASSIFIER_PATH, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
      function_timer = time.time() - function_timer_start
      print('Loading in SVM classifier cost: {}'.format(function_timer))

      if emb_array.shape[0] != 0:
          function_timer_start = time.time()
          predictions = model.predict_proba(emb_array)
          function_timer = time.time() - function_timer_start
          print('Predicting using SVM cost: {}'.format(function_timer))
          best_class_indices = np.argmax(predictions, axis=1)
          best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

          elapsed_inference_time = time.time() - initial_inference_start_time
          print('Total inference time cost: {}'.format(elapsed_inference_time))

          for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

          draw_detection_box(image,ids,bbox_dict,class_names,best_class_indices,best_class_probabilities)
          cv2.imshow('image_view', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break




