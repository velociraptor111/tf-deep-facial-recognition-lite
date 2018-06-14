# tf-deep-facial-recognition-lite
Object Detection and Recognition pipeline using Single Shot Multibox Detector (SSD) with MobileNet as the feature extractor, and FaceNet to recognize faces. The 'lite' in the repository name indicates that this repo was built with the intention of running detection solely on CPU for real time purposes. Using GPU however, will definitely reduce the lag in detecting and recognizing faces for real time tasks.

## Prerequisites
- six==1.9.0
- h5py==2.7.0
- requests==2.18.4
- opencv_python==3.3.0.10
- tensorflow==1.8.0
- scipy==1.0.0
- numpy==1.12.1
- scikit_learn==0.19.1

## Installation

## Notes:
- This repository is still under heavy development.

## TODO:
- Create a download script to download weights from google drive
- Write comments on the different functions
- Configure webcam script to work with GPU (Zain)
- Change the loading of Facenet to grab images directly from numpy array as oppose to saving it on a temp disk

