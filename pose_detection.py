import tensorflow as tf
import cv2

### Tf_Pose Functions ###
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

if __name__ == "__main__":

    with tf.Graph().as_default():
        ### Loading the TF Pose Estimator ###
        w, h = model_wh('432x368')
        e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

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

            humans = e.inference(image, resize_to_default=True, upsample_size=4)

            image_display = TfPoseEstimator.draw_humans(image_display, humans, imgcopy=False)

            cv2.imshow('detect-human-poses', image_display)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


