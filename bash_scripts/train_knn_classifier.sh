python facenet/src/k_nearest_neighbours_classifier.py \
TRAIN \
./datasets/self_images_aligned_160_v4  \
./model/Facenet_Model/20180402-114759/20180402-114759.pb \
./trained_svm_knn_face_models/self_images_neighbours_classifier_v4.pkl \
--batch_size 1000