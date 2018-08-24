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
- It has been found that using SSD with Mobile Net (Trained on WIDER FACE Dataset) for Face Detection as opposed to using MTCNN for Face Detection is faster in events where there contains more than one faces in an image. Moreover, it is also better in creating a tight bounding box.
- This repository is still under heavy development.

## TODO:
- Comment utils.py
- Update the download server to download newer SSD Mobilenet Checkpoint
- Write a python script to automatically train a face classifier using a webcam
- Finish writing the installation script