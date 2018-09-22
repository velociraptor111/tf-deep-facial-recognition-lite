import tensorflow as tf
import cv2

import argparse

### Tf_Pose Functions ###
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

PATH_TO_VIDEO = '/Users/petertanugraha/Desktop/writing_blackboard.mov'
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=int, default=0,
                        help='Set to 0 if no video path is provided. If 1 then will use video path specified.')
    args = parser.parse_args()

    if args.video_path == 0:
        video_path = 0
    else:
        video_path = PATH_TO_VIDEO

    with tf.Graph().as_default():
        ### Loading the TF Pose Estimator ###
        w, h = model_wh('432x368')
        e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

        ### 0 here means start streaming video from webcam
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() is False:
            print("Error opening video stream or file")

        while cap.isOpened():
            _, image = cap.read()
            # Flipping the images in the horizontal direction!
            if args.video_path == 0:
                image = image[..., ::-1, :]

            # This variable is used to be drawn
            image_display = image.copy()

            humans = e.inference(image, resize_to_default=True, upsample_size=4)

            image_display = TfPoseEstimator.draw_humans(image_display, humans, imgcopy=False)

            for human in humans:
                # draw point
                for i in range(common.CocoPart.Background.value):
                    if i in human.body_parts.keys():
                        print(common.coco_part_name_mapping[i])

            cv2.imshow('detect-human-poses', image_display)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


