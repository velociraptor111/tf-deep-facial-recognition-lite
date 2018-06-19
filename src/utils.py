""" Helper Functions written to aid the inference detections """

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

import tensorflow as tf
import cv2
import numpy as np
import facenet
import align.detect_face


def filter_ssd_predictions(dets,threshold=0.7):
  '''
  Takes in a numpy array of detections with its confidence values, and only keeps the detections that
  have confidence value above a certain threshold.

  :param dets: Numpy array with shape (?,6) where index 0 - 3 is bounding box (ymin,xmin,ymax,xmax),
              4 is confidence score and 5 is the class of the prediction
  :param threshold: The confidence threshold which we will keep the predictions
  :return:  Numpy array with the shape (?,6) with the filtered detections based on conf threshold
  '''
  conf_scores = dets[:,4] # Get all the prediction conf scores
  ids = np.where(conf_scores > threshold)
  return dets[ids]
def draw_detection_box(image,ids,bbox_dict,class_names,best_class_indices,best_class_probabilities,threshold=0.7):
  '''
  Draws bounding box on an image using cv2 built in functions.
  :param image: numpy array describing an image that will draw the detection results on
  :param ids: List of detection ids in the correct order
  :param bbox_dict: Dictionary with key the detection id paired with value of bounding box detected by the SSD
  :param class_names: A list of the possible output of the SVM which is trained on
  :param best_class_indices: A list of the output class of the detection. This is referenced to the class_names
  :param best_class_probabilities: List of the probabilities
  :return:
  '''

  for i,id in enumerate(ids):
    bbox = bbox_dict[id]
    cv2.rectangle(image, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (255, 0, 0), 2)
    if best_class_probabilities[i] > threshold:
      cv2.putText(image, class_names[best_class_indices[i]] , (int(bbox[0]), int(bbox[2]) + 10), 0, 0.8, (0, 255, 0),thickness=2)
    else:
      cv2.putText(image, 'Unknown Face', (int(bbox[0]), int(bbox[2]) + 10), 0, 0.8, (0, 0, 255),thickness=2)

def crop_ssd_prediction(xmin,xmax,ymin,ymax,CROP_SSD_PERCENTAGE,im_width,im_height):
  '''
  Cropping the detected ssd bounding box from the given image with an additional specified margin percentage.
  Special care has to be taken to ensure that the cropped box still lies within the image

  :param xmin: x minimum of the bounding box rectangle
  :param xmax: x maximum of the bounding box rectangle
  :param ymin: y minimum of the bounding box rectangle
  :param ymax: x maximum of the bounding box rectangle
  :param CROP_SSD_PERCENTAGE: calculate the percentage of width/height and use this to create a bigger margin
  :param im_width: width of the original image
  :param im_height: height of the original image
  :return: the newly bounding box position of the images
  '''
  xWidth = xmax - xmin
  yHeight = ymax - ymin
  delta_x = int(CROP_SSD_PERCENTAGE * xWidth)
  delta_y = int(CROP_SSD_PERCENTAGE * yHeight)
  new_xmin = max(0, int(xmin - delta_x))
  new_xmax = min(int(xmax + delta_x), im_width)
  new_ymin = max(0, int(ymin - delta_y))
  new_ymax = min(int(ymax + delta_y), im_height)

  return new_xmin,new_xmax,new_ymin,new_ymax

def post_process_ssd_predictions(boxes,scores,classes):
  '''
  Combines the numpy arrrays given by the Tensorflow Object Detection API into a single det array
  for further processing.

  :param boxes:
  :param scores:
  :param classes:
  :return:
  '''
  boxes = np.squeeze(boxes)
  scores = np.reshape(scores, (scores.shape[1], scores.shape[0]))
  classes = np.reshape(classes, (classes.shape[1], classes.shape[0]))

  dets = np.hstack((boxes, scores))
  dets = np.hstack((dets, classes))
  dets = filter_ssd_predictions(dets, threshold=0.7)
  return dets

def load_tf_ssd_detection_graph(PATH_TO_CKPT):
  '''
  Loads the Tensorflow SSD Object Detection API into memory

  :param PATH_TO_CKPT:
  :return:
  '''
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
  boxes_tensor = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
  scores_tensor = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
  classes_tensor = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
  num_detections_tensor = tf.get_default_graph().get_tensor_by_name('num_detections:0')

  return image_tensor,boxes_tensor,scores_tensor,classes_tensor,num_detections_tensor

def load_tf_facenet_graph(FACENET_MODEL_PATH):
  '''
  Loads the Facenet Tensorflow Graph into memory.

  :param FACENET_MODEL_PATH:
  :return:
  '''

  facenet.load_model(FACENET_MODEL_PATH)
  images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
  embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
  phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

  return images_placeholder,embeddings,phase_train_placeholder

def print_recognition_output(best_class_indices, class_names, best_class_probabilities,recognition_threshold=0.7):
  '''
  Prints the output class of the detection. If it is below a certain threshold, then it will display it as an unknown
  class.

  :param best_class_indices:
  :param class_names:
  :param best_class_probabilities:
  :param recognition_threshold:
  :return:
  '''
  for i in range(len(best_class_indices)):
    if best_class_probabilities[i] > recognition_threshold:
      print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
    else:
      print('%4d  %s: %.3f' % (i, 'Unknown Face', best_class_probabilities[i]))

def prewhiten(x):
  """

  :param x: The numpy array representing an image
  :return: A whitened image
  """
  mean = np.mean(x)
  std = np.std(x)
  std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
  y = np.multiply(np.subtract(x, mean), 1/std_adj)
  return y

def get_face_embeddings(sess,embeddings,images_placeholder,phase_train_placeholder,
                        nrof_images,nrof_batches_per_epoch,FACENET_PREDICTION_BATCH_SIZE,images_array):
  """

  :param sess: Current Tensorflow session variable
  :param embeddings: A tensor variable that holds the embeddings of the result
  :param images_placeholder: A tensor variable that holds the images
  :param phase_train_placeholder: A tensor variable
  :param nrof_images: Number of detected faces
  :param nrof_batches_per_epoch: Number of images to run per epoch
  :param FACENET_PREDICTION_BATCH_SIZE: Number of maximum faces per facenet detection
  :param images_array: Numpy representation of an image.
  :return:
  """
  embedding_size = embeddings.get_shape()[1]
  emb_array = np.zeros((nrof_images, embedding_size))

  for i in range(nrof_batches_per_epoch):
    start_index = i * FACENET_PREDICTION_BATCH_SIZE
    end_index = min((i + 1) * FACENET_PREDICTION_BATCH_SIZE, nrof_images)
    images_batch = images_array[start_index:end_index]  # Pass in several different paths
    images_batch = np.array(images_batch)
    feed_dict = {images_placeholder: images_batch, phase_train_placeholder: False}
    function_timer_start = time.time()
    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    function_timer = time.time() - function_timer_start
    print('Calculating image embedding cost: {}'.format(function_timer))

  return emb_array
