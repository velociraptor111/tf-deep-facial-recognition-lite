python facenet/src/classifier.py \
TRAIN \
./datasets/self_images_aligned_160_v4 \
./model/Facenet_Model/20180402-114759/20180402-114759.pb \
./trained_svm_knn_face_models/self_images_classifier_v4.pkl \
--batch_size 1000