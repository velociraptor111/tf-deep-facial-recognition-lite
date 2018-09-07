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
import math
import pickle

import numpy as np
import tensorflow as tf
import cv2

'''
    Load facenet/src directory to system path
'''
import _init_paths

### Facenet  ###
from align.detect_face import create_mtcnn

### Utility Functions ###
from src.align_image_mtcnn import align_image_with_mtcnn_with_tf_graph
from src.utils import load_tf_ssd_detection_graph,run_inference_for_single_image_through_ssd,post_process_ssd_predictions \
    ,load_tf_facenet_graph,crop_ssd_prediction,prewhiten,get_face_embeddings,check_face_bbox_inside,\
    get_body_keypoints_centroid,check_pose_centroid_inside

### Tf_Pose Functions ###
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = config.get("DEFAULT","PATH_TO_SSD_CKPT")
FACENET_MODEL_PATH = config.get("DEFAULT","PATH_TO_FACENET_MODEL")
IMAGE_SIZE = int(config.get("DEFAULT","IMAGE_SIZE"))
FACENET_PREDICTION_BATCH_SIZE = int(config.get("DEFAULT","FACENET_PREDICTION_BATCH_SIZE"))

CLASSIFIER_PATH_SVM = './trained_svm_knn_face_models/self_images_classifier_v4.pkl'
CLASSIFIER_PATH_KNN = './trained_svm_knn_face_models/self_images_neighbours_classifier_v4.pkl'
PATH_TO_PERSON_PB = './model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'

class Person:
    def __init__(self,id_name):
        self.name = id_name
        self.face_bbox = []
        self.body_bbox = []
        self.body_pose = {}

        if id_name != 'Unknown':
            color_pallete = list(np.random.choice(range(256), size=3))
            color_pallete = (int(color_pallete[0]), int(color_pallete[1]), int(color_pallete[2]))
        else:
            color_pallete = (0,0,255)
        self.color_pallete = color_pallete


if __name__ == "__main__":

    # Global variable that will hold all the "Recognized Human in the whole lifetime of Videoing"
    Identified_Human_List = []
    with tf.Graph().as_default():

        ### Loading Face Detector ###
        Face_Detection_Graph = tf.Graph()
        image_tensor, tensor_dict = load_tf_ssd_detection_graph(PATH_TO_CKPT, input_graph=Face_Detection_Graph)
        face_detection_sess = tf.Session(graph=Face_Detection_Graph)

        ### Loading Person Detector ###
        person_image_tensor, person_tensor_dict = load_tf_ssd_detection_graph(PATH_TO_PERSON_PB, input_graph=None)
        main_sess = tf.Session()

        ### Loading the SVM Classifier for Face ID classification ###
        with open(CLASSIFIER_PATH_SVM, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        ### Loading the KNN Classifier for Face Recognition Classifier ###
        with open(CLASSIFIER_PATH_KNN, 'rb') as infile:
            knn_model = pickle.load(infile)

        ### Loading the TF Pose Estimator ###
        w, h = model_wh('432x368')
        e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

        with main_sess.as_default():
            ### Creating and Loading MTCNN ###
            pnet, rnet, onet = create_mtcnn(main_sess, None)
            ### Creating and Loading the Facenet Graph ###
            images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)

        ### 0 here means start streaming video from webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened() is False:
            print("Error opening video stream or file")

        while cap.isOpened():
            _, image = cap.read()
            # Convert image from BGR to RGB color scheme
            image = image[..., ::-1, :]
            # This variable is used to be drawn
            image_display = image.copy()

            # Posenet INFERENCE
            humans = e.inference(image, resize_to_default=True, upsample_size=4)

            # This list will store the centroid information of the collected pose
            human_centroid_array = []
            for human in humans:
                centroid_tuple = get_body_keypoints_centroid(human,image)
                cv2.circle(image_display, centroid_tuple, 3, [255,0,0], thickness=3, lineType=8, shift=0)
                human_centroid_array.append(centroid_tuple)


            initial_inference_start_time = time.time()
            # Both the SSD and Facenet also uses np.uint8 and RGB images for both!
            image_np = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.uint8)

            with face_detection_sess.as_default():
                # Face Detection INFERENCE
                output_dict = run_inference_for_single_image_through_ssd(face_detection_sess, image_np, image_tensor, tensor_dict)

            with main_sess.as_default():
                # Body Detection INFERENCE
                output_dict_person = run_inference_for_single_image_through_ssd(main_sess, image_np, person_image_tensor, person_tensor_dict)

                # Post process the Face and Body results
                face_dets = post_process_ssd_predictions(image_np, output_dict, threshold=0.25)
                person_dets = post_process_ssd_predictions(image_np, output_dict_person, threshold=0.5,detection_classes = [1])

                # A dict variable used to store the Face detection ID and its bounding box values
                faces_bbox_dict = {}
                face_ids = []
                images_array = []

                for detection_id, cur_det in enumerate(face_dets):
                    boxes = cur_det[:4]
                    (ymin, xmin, ymax, xmax) = (boxes[0], boxes[1],
                                                boxes[2], boxes[3])
                    bbox = (xmin, xmax, ymin, ymax)
                    new_xmin, new_xmax, new_ymin, new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax,
                                                                                 None, image_np.shape[1],
                                                                                 image_np.shape[0],0.5,0.3)
                    roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]

                    ### Need to resize to 250 by 250 first here before aligning with mtcnn
                    roi_cropped_rgb = cv2.resize(roi_cropped_rgb, (250, 250))

                    faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb, pnet, rnet, onet,
                                                                        image_size=IMAGE_SIZE)

                    if len(faces_roi) != 0:  # This is either a face or not a face
                        faces_roi = faces_roi[0]
                        images_array.append(prewhiten(faces_roi))
                        face_ids.append(detection_id)
                        faces_bbox_dict[detection_id] = bbox

                nrof_faces = len(face_ids)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_faces / FACENET_PREDICTION_BATCH_SIZE))
                face_embedding_array = get_face_embeddings(main_sess, embeddings, images_placeholder, phase_train_placeholder,
                                                nrof_faces, nrof_batches_per_epoch, FACENET_PREDICTION_BATCH_SIZE,
                                                images_array)


                '''
                    In the tree diagram this is the branch that corresponds to that it has detected a face in the frame. 
                    Although we do not know if this is the Person of Interest or could just be a random student passing by frame.
                    The necessary steps below then tries to Identify(ID) and Recognize the detected face using Facial Recognition.
                    
                    TODO: Need to be able to still ID the Professor even if he/she turns his back on the camera.
                '''
                if face_embedding_array.shape[0] != 0:
                    print("Detected at least one Face")

                    # Running INFERENCE on KNN and SVM models
                    distances, indices = knn_model.kneighbors(face_embedding_array)
                    predictions = model.predict_proba(face_embedding_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    average_distance_array = np.mean(distances,axis=1)

                    # This array holds all the ID'ed human detected at the current image frame
                    single_frame_human_list = []
                    for i,id in enumerate(face_ids):
                        bbox = faces_bbox_dict[id]

                        '''
                            The euclidean distance of the current face vector is less than a threshold of the training set. 
                            Meaning it has seen the face previously before ...
                        '''
                        if average_distance_array[i] < 0.85:
                            found_recognized_human_bool = False
                            human_obj = None

                            ### Check first for any human in the Identified_Human_List .. This will usually be just the Professor .. Unless there is more than one Prof ###
                            for cur_human_obj in Identified_Human_List:
                                if class_names[best_class_indices[i]] == cur_human_obj.name:
                                    cur_human_obj.face_bbox=bbox ### Update the facebbox
                                    human_obj = cur_human_obj
                                    found_recognized_human_bool = True

                            if not found_recognized_human_bool:
                                # Create a new human object and append to the human list
                                print("Creating a new human object . His name is: ", class_names[best_class_indices[i]])
                                human_obj = Person(class_names[best_class_indices[i]])
                                human_obj.face_bbox = bbox
                                Identified_Human_List.append(human_obj)

                        else:
                            human_obj = Person("Unknown")
                            human_obj.face_bbox = bbox

                        ### Associating the detected BODY array with the current ID'ed Person ###
                        for single_person_detection in person_dets:
                            if check_face_bbox_inside(bbox,single_person_detection):
                                # Doing this will overwrite the body_bbox detection if there are multiple detections for a single frame
                                human_obj.body_bbox = single_person_detection

                        ### Associating the detected POSES with the current person ####
                        for idx,body_pose_centroid in enumerate(human_centroid_array):
                            if len(human_obj.body_bbox) != 0: # Sometimes the person may have a Face but no Pose for this frame ... ###
                                if check_pose_centroid_inside(body_pose_centroid,human_obj.body_bbox):
                                    human_obj.body_pose = humans[idx]

                        single_frame_human_list.append(human_obj)

                    print("There are ", len(single_frame_human_list) , "human beings in the frame that can be identified.")

                    '''
                        Will only display on screen if Face is Detected and Recognized, Body is Detected and Pose is Detected
                    '''
                    for human_obj in single_frame_human_list:
                        if len(human_obj.body_bbox) != 0 and len(human_obj.face_bbox) != 0 and bool(human_obj.body_pose) != False :
                            color_pallete = human_obj.color_pallete
                            # Draw the human pose
                            image_display = TfPoseEstimator.draw_humans(image_display, [human_obj.body_pose], imgcopy=False)
                            # Draw the body
                            cv2.rectangle(image_display, (int(human_obj.body_bbox[1]), int(human_obj.body_bbox[0])), (int(human_obj.body_bbox[3]), int(human_obj.body_bbox[2])),color_pallete, 2)
                            # Draw the face
                            cv2.rectangle(image_display, (int(human_obj.face_bbox[0]), int(human_obj.face_bbox[2])), (int(human_obj.face_bbox[1]), int(human_obj.face_bbox[3])),color_pallete, 2)
                            # Write person name
                            cv2.putText(image_display, str(human_obj.name), (int(human_obj.face_bbox[0]), int(human_obj.face_bbox[2]) + 10), 0, 0.8,color_pallete,thickness=2)

                cv2.imshow('full-face-detection-pipeline', image_display)
                if cv2.waitKey(1) == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()





