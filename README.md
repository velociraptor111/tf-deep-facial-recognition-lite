# tf-deep-facial-recognition-lite
Object Detection and Recognition pipeline using Single Shot Multibox Detector (SSD) with MobileNet as the feature extractor, and FaceNet to recognize faces. The 'lite' in the repository name indicates that this repo was built with the intention of running detection solely on CPU for real time purposes. Using GPU however, will definitely reduce the lag in detecting and recognizing faces for real time tasks.

## Prerequisites
Tested with Python3 only. 
It is HIGHLY recommended that you create a fresh new conda environment, and use anaconda for managing the python packages.
- psutil==5.4.6
- tqdm==4.24.0
- numpy==1.14.3
- tensorpack==0.8.8
- h5py==2.8.0
- requests==2.19.1
- setuptools==40.0.0
- six==1.11.0
- scipy==1.1.0
- tensorflow==1.9.0
- opencv_python==3.4.2.17
- matplotlib==2.2.3
- protobuf==3.6.1
- scikit_learn==0.19.2

## Installation
```buildoutcfg
git clone https://github.com/velociraptor111/tf-deep-facial-recognition-lite.git
cd tf-deep-facial-recognition-lite
pip install requirements.txt
python download_model_checkpoints.py
```
## Demo Scripts
- Full pipeline
```buildoutcfg
python combined_inferences.py
```
- OpenPose (Mobilenet-Thin)
```buildoutcfg
python pose_detection.py
```
- Face Detection (SSD with MobileNet trained on OpenFace)
```buildoutcfg
python face_detection_with_ssd_video.py
```
- Body Detection (SSD with MobileNet trained on COCO Dataset)
```buildoutcfg
python human_detection.py
```

## Notes:
- It has been found that using SSD with Mobile Net (Trained on WIDER FACE Dataset) for Face Detection as opposed to using MTCNN for Face Detection is faster in events where there contains more than one faces in an image. Moreover, it is also better in creating a tight bounding box.
- This repository is still under heavy development.

## TODO:
- Finalize the pipeline scripts
- Create RNN to classify OpenPose features for gesture recognition
- Human Identification by extracting clothing features