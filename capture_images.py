import numpy as np
import cv2
import os
import tensorflow as tf
from src.utils import post_process_ssd_predictions,run_inference_for_single_image,load_tf_ssd_detection_graph,crop_ssd_prediction

TARGET_DIRECTORY = './datasets/self_images'
PERSON_NAME = 'John_Doe'
TARGET_DIRECTORY = os.path.join(TARGET_DIRECTORY,PERSON_NAME)


if not os.path.isdir(TARGET_DIRECTORY):
    os.makedirs(TARGET_DIRECTORY)

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def run_tf_object_detection_video(input_graph,image_tensor, tensor_dict,PICTURE_ID,path_to_video=None):

  if path_to_video == None:
    cap = cv2.VideoCapture(0)
  else:
    cap = cv2.VideoCapture(path_to_video)

  if cap.isOpened() is False:
    print("Error opening video stream or file")

  with input_graph.as_default():
    with tf.Session() as sess:
      while cap.isOpened():
        _, image = cap.read()
        image = image[..., ::-1, :]
        image_display = image.copy() # This variable is for the purpose of displaying the picture to the user

        image_np = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.uint8) #Convert to RGB and convert to uint8
        output_dict = run_inference_for_single_image(sess,image_np,image_tensor,tensor_dict)
        dets = post_process_ssd_predictions(image_np,output_dict,threshold=0.5)

        for cur_det in dets:
          boxes = cur_det[:4]
          ymin = boxes[0]
          xmin = boxes[1]
          ymax = boxes[2]
          xmax = boxes[3]
          conf_score = cur_det[4]
          # This is still RGB here,that's why the first element is Red
          cv2.rectangle(image_display,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)

          new_xmin, new_xmax, new_ymin, new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax,None,image_np.shape[1], image_np.shape[0],0.5,0.3)
          cropped_coordinates = (new_xmin, new_ymin, new_xmax, new_ymax)
          cv2.rectangle(image_display, (int(new_xmin), int(new_ymin)), (int(new_xmax), int(new_ymax)), (255, 0, 255), 3)

        if dets.shape[0] > 1:
            cv2.putText(image_display, 'Face is not stable or there may be more than one person in camera. Please readjust accordingly', (0,20), 0, 0.7, (0, 0, 255), thickness=2)
        else:
            cv2.putText(image_display,
                        'Please make sure you have a good lighting.',
                        (0, 20), 0, 0.7, (0, 255, 0), thickness=2)


        cv2.imshow('face-detection-ssd', image_display)

        if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
            if dets.shape[0] == 1:
                cropped_img = image[cropped_coordinates[1]:cropped_coordinates[3], cropped_coordinates[0]:cropped_coordinates[2]]
                resized_frame = cv2.resize(cropped_img, (250, 250))

                resized_frame_gamma_0_5 = adjust_gamma(resized_frame, gamma=0.5)
                resized_frame_gamma_0_7 = adjust_gamma(resized_frame, gamma=0.7)
                resized_frame_gamma_1_5 = adjust_gamma(resized_frame, gamma=1.5)
                resized_frame_gamma_1_8 = adjust_gamma(resized_frame, gamma=1.8)

                cv2.imwrite(os.path.join(TARGET_DIRECTORY,'image_'+ str(PICTURE_ID)+'.jpg'),resized_frame)
                cv2.imwrite(os.path.join(TARGET_DIRECTORY, 'image_' + str(PICTURE_ID) + '_gamma' + str(0.4) + '.jpg'),
                            resized_frame_gamma_0_5)
                cv2.imwrite(os.path.join(TARGET_DIRECTORY, 'image_' + str(PICTURE_ID) + '_gamma' + str(1.8) + '.jpg'),
                            resized_frame_gamma_1_8)
                cv2.imwrite(os.path.join(TARGET_DIRECTORY, 'image_' + str(PICTURE_ID) + '_gamma'+ str(0.7) + '.jpg'), resized_frame_gamma_0_7)
                cv2.imwrite(os.path.join(TARGET_DIRECTORY, 'image_' + str(PICTURE_ID) + '_gamma'+ str(1.5)  + '.jpg'), resized_frame_gamma_1_5)

                print("Image ",str(PICTURE_ID),"has been saved along with different gamma preprocessing!")
                PICTURE_ID += 1
            else:
                print("Picture is not stable. Found two detections. Please readjust until only one face is detected!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

  cap.release()
  cv2.destroyAllWindows()
  return

if __name__ == "__main__":
  PATH_TO_FROZEN_GRAPH = '/Users/petertanugraha/Projects/tf-deep-facial-recognition-lite/model/ssd_mobilenet_v1_focal_loss_face_mark_2.pb'
  PICTURE_ID = 0

  main_graph = tf.Graph()
  image_tensor,tensor_dict=load_tf_ssd_detection_graph(PATH_TO_FROZEN_GRAPH,input_graph=main_graph)
  run_tf_object_detection_video(main_graph,image_tensor,tensor_dict,PICTURE_ID,path_to_video=None)