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
#Very important to note here that the facenet.py file imported here will be dependant on the init_path supply
# of where the facenet source code to extract from! *Ps. script will break when you have manually exported the facenet/src directory ..
import facenet

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

def crop_ssd_prediction(xmin,xmax,ymin,ymax,CROP_SSD_PERCENTAGE,im_width,im_height,crop_ssd_percentage_x=None,crop_ssd_percentage_y=None):
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

  if crop_ssd_percentage_x == None and crop_ssd_percentage_y == None:
    delta_x = int(CROP_SSD_PERCENTAGE * xWidth)
    delta_y = int(CROP_SSD_PERCENTAGE * yHeight)
  else:
    delta_x = int(crop_ssd_percentage_x * xWidth)
    delta_y = int(crop_ssd_percentage_y * yHeight)

  new_xmin = max(0, int(xmin - delta_x))
  new_xmax = min(int(xmax + delta_x), im_width)
  new_ymin = max(0, int(ymin - delta_y))
  new_ymax = min(int(ymax + delta_y), im_height)

  return new_xmin,new_xmax,new_ymin,new_ymax

def post_process_ssd_predictions(image_np,output_dict,threshold=0.3,detection_classes=None):
  '''
  Filter results of output from Single Shot Detectors. Mainly only keeping detection values with confidence above a
  certain threshold and also filtering the class of the detections.
  :param image_np: Multi dimensional image array
  :param output_dict: Output dictionary produced from the Single Shot Detector
  :param threshold: The minimum confidence value to keep detections
  :param detection_classes: The class of the detection (e.g A human, chair,sofa , etc.)
  :return: Squashed filtered array
  '''
  image_width = image_np.shape[1]
  image_height = image_np.shape[0]

  if detection_classes == None:
    boxes = np.squeeze(output_dict['detection_boxes'])
    boxes[:,0] = boxes[:,0] * image_height
    boxes[:,1] = boxes[:,1] * image_width
    boxes[:,2] = boxes[:,2] * image_height
    boxes[:,3] = boxes[:,3] * image_width
    scores = np.reshape(output_dict['detection_scores'], (output_dict['detection_scores'].shape[0], 1))
    dets = np.hstack((boxes, scores))
  else:
    final_class_filtered_index = []
    for cur_class in detection_classes:
      final_class_filtered_index.append(np.where(output_dict['detection_classes'] == cur_class)[0])

    boxes = np.squeeze(output_dict['detection_boxes'])
    boxes = boxes[final_class_filtered_index]
    boxes[:, 0] = boxes[:, 0] * image_height
    boxes[:, 1] = boxes[:, 1] * image_width
    boxes[:, 2] = boxes[:, 2] * image_height
    boxes[:, 3] = boxes[:, 3] * image_width

    conf_scores = output_dict['detection_scores']
    conf_scores = conf_scores[final_class_filtered_index]
    scores = np.reshape(conf_scores, (conf_scores.shape[0], 1))
    dets = np.hstack((boxes, scores))

  dets = filter_ssd_predictions(dets,threshold)

  return dets


def run_inference_for_single_image_through_ssd(sess,image,image_tensor,tensor_dict):
  '''
  Do a single inference of an image passing through a Single Shot Detector.
  :param sess: A Tensorflow Session object. Session should contain the respective Tensorflow Graph that is being inferred.
  :param image: Multi dimensional numpy array
  :param image_tensor: Tensor variable used to receive in the images
  :param tensor_dict: Dict containing the output values
  :return: Dict with detections
  '''

  output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict





def load_tf_ssd_detection_graph(PATH_TO_FROZEN_GRAPH,input_graph=None):
  '''
  Loading the SSD graph into memory. If given an input graph, will put the SSD into that specified graph,
  else will grab the current graph using tf.get_default_graph().
  Right now there are only two graph, the "Main" and "Face Detection Graph"
  :param PATH_TO_FROZEN_GRAPH: String variable which contains path to the saved frozen graph file.
  :param input_graph: Tf.Graph variable
  :return: Returns Tensorflow variables which will be supplied to sess.run later on when run for inference.
  '''

  if input_graph == None:
      current_graph = tf.get_default_graph()
  else:
      current_graph = input_graph

  with current_graph.as_default(): #Doing this will add whatever operation below into my current maing raph
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    return image_tensor, tensor_dict

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

def check_face_bbox_inside(bbox, single_person_detection):
  '''
  Check if the bounding box for face lies within the bounding box for the Human Body
  :param bbox: Tuple (xmin,xmax,ymin,ymax)
  :param single_person_detection: Tuple (ymin,xmin,ymax,xmax)
  :return: boolean
  '''
  body_ymin = single_person_detection[0]
  body_xmin = single_person_detection[1]
  body_ymax = single_person_detection[2]
  body_xmax = single_person_detection[3]

  if bbox[0] > body_xmin and bbox[0] < body_xmax:
    if bbox[1] > body_xmin and bbox[1] < body_xmax:
      if bbox[2] > body_ymin and bbox[2] < body_ymax:
        if bbox[3] > body_ymin and bbox[3] < body_ymax:
          return True

  return False


def get_body_keypoints_centroid(human, image):
  '''
  Return a tuple of int values containing x and y position values for centroid respectively.
  :param human: A Human object defined in tf_pose/estimator
  :param image: Numpy array
  :return: Tuple
  '''

  image_height = image.shape[0]
  image_width = image.shape[1]
  # Need to create a list of tuples containing the (x,y) values for each keypoints
  keypoints_list = []
  for key_point_id in human.body_parts.keys():
    keypoints_list.append([int(human.body_parts[key_point_id].x * image_width + 0.5),
                           int(human.body_parts[key_point_id].y * image_height + 0.5)])
  keypoints_list = np.array(keypoints_list)
  total_body_parts = len(human.body_parts.keys())
  sum_x = np.sum(keypoints_list[:, 0])
  sum_y = np.sum(keypoints_list[:, 1])
  return int(sum_x / total_body_parts), int(sum_y / total_body_parts)


def check_pose_centroid_inside(centroid, single_person_detection):
  '''
  Return True if the centroid location is within the bounding box of the body.
  :param centroid: Tuple (x,y)
  :param single_person_detection: Tuple (ymin,xmin,ymax,xmax)
  :return: Boolean
  '''
  body_ymin = single_person_detection[0]
  body_xmin = single_person_detection[1]
  body_ymax = single_person_detection[2]
  body_xmax = single_person_detection[3]

  if centroid[0] > body_xmin and centroid[0] < body_xmax:
    if centroid[1] > body_ymin and centroid[1] < body_ymax:
      return True

  return False